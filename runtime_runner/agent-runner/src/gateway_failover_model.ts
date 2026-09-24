/**
 * Turn-level recovery for OpenAI-compatible gateways that fail over between
 * upstream accounts.
 *
 * Reasoning models return encrypted reasoning items that the Agents SDK
 * replays on every later turn. Only the upstream account that produced them
 * can decrypt them, so when a gateway routes a turn to a different account
 * the request is rejected with a 400 ("encrypted content for item ... could
 * not be verified", "encrypted conversation context could not be
 * validated", "conversation context is not compatible with the available
 * resources"). Without recovery the whole agent run fails and the task or
 * forum agent is retried from scratch, or is lost once retries run out.
 *
 * On exactly those errors this wrapper re-sends the same turn once with the
 * replayed reasoning items removed and item ids dropped (so every item is
 * sent by value). Tool calls, tool outputs and messages are kept, so the
 * conversation is intact; only the model's hidden reasoning from earlier
 * turns is missing on that one turn. Requests that succeed are untouched.
 */

type Logger = (message: string) => void;

const GATEWAY_STATE_ERROR_MARKERS = [
  'encrypted content for item',
  'encrypted conversation context could not be validated',
  'conversation context is not compatible with the available resources',
];

export function isGatewayStateError(err: unknown): boolean {
  const text = String((err as any)?.message ?? err ?? '').toLowerCase();
  return GATEWAY_STATE_ERROR_MARKERS.some((marker) => text.includes(marker));
}

/** Drop replayed reasoning items and item ids so the turn is self-contained. */
export function stripGatewayState<T>(input: T): T {
  if (!Array.isArray(input)) return input;
  return input
    .filter((item) => (item as any)?.type !== 'reasoning')
    .map((item) => {
      if (!item || typeof item !== 'object' || !('id' in (item as any))) return item;
      const { id: _id, ...rest } = item as any;
      return rest;
    }) as unknown as T;
}

interface ModelLike {
  getResponse(request: any): Promise<any>;
  getStreamedResponse(request: any): AsyncIterable<any>;
}

export class GatewayFailoverModel implements ModelLike {
  recoveries = 0;

  constructor(
    private readonly inner: ModelLike,
    private readonly log: Logger = () => {},
  ) {}

  async getResponse(request: any): Promise<any> {
    try {
      return await this.inner.getResponse(request);
    } catch (err) {
      if (!isGatewayStateError(err) || !Array.isArray(request?.input)) throw err;
      this.recoveries += 1;
      this.log(
        `gateway state error; re-sending turn without replayed reasoning (recovery #${this.recoveries})`,
      );
      return this.inner.getResponse({ ...request, input: stripGatewayState(request.input) });
    }
  }

  getStreamedResponse(request: any): AsyncIterable<any> {
    return this.inner.getStreamedResponse(request);
  }
}
