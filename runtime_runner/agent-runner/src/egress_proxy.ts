import http from 'node:http';
import net from 'node:net';

/** Only provider API ports are normally needed. */
const ALLOWED_PORTS = new Set([443, 80]);

// A locally hosted Responses-to-Chat adapter is the one deliberate exception:
// it is reachable only through Docker Desktop's reserved hostname and lets a
// Chat-Completions-only provider work with the KSI OpenAI Responses runner.
// Keep this a fixed host+port pair rather than opening an arbitrary high port.
const LOCAL_COMPAT_PROXY_HOST = 'host.docker.internal';
const LOCAL_COMPAT_PROXY_PORT = 4002;

/** Exact-hostname, case-insensitive match. No wildcard/suffix matching by
 *  design — suffix matching would allow `api.anthropic.com.evil.com`. */
export function isAllowed(
  host: string,
  port: number,
  allowlist: ReadonlySet<string>,
  allowedPorts: ReadonlySet<number> = ALLOWED_PORTS,
): boolean {
  if (host.toLowerCase() === LOCAL_COMPAT_PROXY_HOST && port === LOCAL_COMPAT_PROXY_PORT) {
    return allowlist.has(LOCAL_COMPAT_PROXY_HOST);
  }
  if (!allowedPorts.has(port)) return false;
  return allowlist.has(host.toLowerCase());
}

export function parseAllowlist(raw: string | undefined): Set<string> {
  return new Set(
    (raw || '')
      .split(',')
      .map((h) => h.trim().toLowerCase())
      .filter((h) => h.length > 0),
  );
}

/** A CONNECT-only forward proxy. Plain HTTP proxying is rejected; only
 *  allowlisted CONNECT tunnels are established.
 *
 *  @param allowedPorts  @internal — test seam; production callers use the
 *                       default 443/80. Lets tests allow an ephemeral port
 *                       without needing root. */
export function createEgressProxy(
  allowlist: ReadonlySet<string>,
  allowedPorts: ReadonlySet<number> = ALLOWED_PORTS,
): http.Server {
  const server = http.createServer((_req, res) => {
    res.writeHead(405, { 'content-type': 'text/plain' });
    res.end('egress proxy: only CONNECT is supported\n');
  });

  server.on('connect', (req, clientSocket, head) => {
    const target = req.url || '';
    const lastColon = target.lastIndexOf(':');
    const host = lastColon > 0 ? target.slice(0, lastColon) : '';
    const port = lastColon > 0 ? Number(target.slice(lastColon + 1)) : NaN;

    if (!host || !Number.isInteger(port) || !isAllowed(host, port, allowlist, allowedPorts)) {
      clientSocket.write('HTTP/1.1 403 Forbidden\r\n\r\n');
      clientSocket.destroy();
      return;
    }

    const upstream = net.connect(port, host, () => {
      clientSocket.write('HTTP/1.1 200 Connection Established\r\n\r\n');
      upstream.write(head);
      upstream.pipe(clientSocket);
      clientSocket.pipe(upstream);
    });
    upstream.on('error', () => clientSocket.destroy());
    clientSocket.on('error', () => upstream.destroy());
  });

  return server;
}
