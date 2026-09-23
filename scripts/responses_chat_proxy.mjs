#!/usr/bin/env node
/**
 * Local Responses API -> Chat Completions compatibility proxy.
 *
 * KSI's OpenAI agent runtime uses the Responses API (including function calls),
 * while some OpenAI-compatible providers expose only /chat/completions. This
 * process keeps the provider credential on the host and translates the small
 * Responses subset used by @openai/agents. It deliberately listens on a local
 * development port; containers reach it through host.docker.internal.
 */
import http from 'node:http';
import crypto from 'node:crypto';

const upstreamBase = (process.env.URL || '').replace(/\/$/, '');
const upstreamKey = process.env.KEY || '';
const port = Number(process.env.KSI_RESPONSES_PROXY_PORT || 4002);
const maxOutputTokens = Number(process.env.KSI_RESPONSES_PROXY_MAX_TOKENS || 1024);
const conversations = new Map();

if (!upstreamBase || !upstreamKey) {
  throw new Error('URL and KEY must be loaded from the parent .env before starting this proxy.');
}

function json(res, status, body) {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => { raw += chunk; });
    req.on('end', () => {
      try { resolve(raw ? JSON.parse(raw) : {}); } catch (error) { reject(error); }
    });
    req.on('error', reject);
  });
}

function contentText(value) {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return value == null ? '' : JSON.stringify(value);
  return value.map((part) => {
    if (typeof part === 'string') return part;
    if (!part || typeof part !== 'object') return String(part ?? '');
    return part.text ?? part.content ?? '';
  }).join('');
}

function inputToMessages(input) {
  const items = Array.isArray(input) ? input : [input];
  const messages = [];
  for (const item of items) {
    if (typeof item === 'string') {
      messages.push({ role: 'user', content: item });
      continue;
    }
    if (!item || typeof item !== 'object') continue;
    if (item.type === 'function_call_output') {
      messages.push({
        role: 'tool',
        tool_call_id: item.call_id,
        content: typeof item.output === 'string' ? item.output : JSON.stringify(item.output ?? ''),
      });
      continue;
    }
    if (item.type === 'function_call') {
      messages.push({
        role: 'assistant',
        content: '',
        tool_calls: [{
          id: item.call_id,
          type: 'function',
          function: { name: item.name, arguments: item.arguments || '{}' },
        }],
      });
      continue;
    }
    if (item.type === 'message' || item.role) {
      messages.push({ role: item.role === 'assistant' ? 'assistant' : 'user', content: contentText(item.content) });
    }
  }
  return messages;
}

function responseTools(tools) {
  if (!Array.isArray(tools)) return undefined;
  const converted = tools
    .filter((tool) => tool && tool.type === 'function')
    .map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        ...(tool.description ? { description: tool.description } : {}),
        parameters: tool.parameters || { type: 'object', properties: {} },
      },
    }));
  return converted.length ? converted : undefined;
}

async function upstreamChat(body) {
  const response = await fetch(`${upstreamBase}/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${upstreamKey}`, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({ error: { message: 'Provider returned a non-JSON response.' } }));
  if (!response.ok) {
    const error = new Error(payload?.error?.message || `Provider HTTP ${response.status}`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

async function handleResponses(req, res) {
  let request;
  try { request = await readJson(req); } catch { return json(res, 400, { error: { message: 'Invalid JSON request body.' } }); }
  if (request.stream) return json(res, 400, { error: { message: 'Streaming Responses requests are not supported by this local compatibility proxy.' } });
  const previous = request.previous_response_id;
  const prior = previous ? conversations.get(previous) : undefined;
  if (previous && !prior) return json(res, 400, { error: { message: `Unknown previous_response_id: ${previous}` } });

  const messages = prior ? [...prior] : [];
  if (!prior && request.instructions) messages.push({ role: 'system', content: request.instructions });
  messages.push(...inputToMessages(request.input));
  const maxTokens = Number(request.max_output_tokens);
  const chatRequest = {
    model: request.model,
    messages,
    ...(responseTools(request.tools) ? { tools: responseTools(request.tools) } : {}),
    ...(request.tool_choice ? { tool_choice: request.tool_choice } : {}),
    ...(typeof request.parallel_tool_calls === 'boolean' ? { parallel_tool_calls: request.parallel_tool_calls } : {}),
    ...(Number.isFinite(maxTokens) && maxTokens > 0
      ? { max_tokens: Math.min(Math.floor(maxTokens), maxOutputTokens) }
      : { max_tokens: maxOutputTokens }),
    ...(typeof request.temperature === 'number' ? { temperature: request.temperature } : {}),
    ...(typeof request.top_p === 'number' ? { top_p: request.top_p } : {}),
    // KSI uses tools in tight loops. For Chat-Completions reasoning models,
    // suppressing visible chain-of-thought keeps each tool turn bounded and
    // lets the agent act instead of exhausting its turn on deliberation.
    reasoning_effort: 'none',
  };
  let chat;
  try { chat = await upstreamChat(chatRequest); }
  catch (error) {
    return json(res, error.status || 502, error.payload || { error: { message: String(error.message || error) } });
  }
  const choice = chat.choices?.[0] || { message: {} };
  const message = choice.message || {};
  const responseId = `resp_${crypto.randomUUID().replaceAll('-', '')}`;
  const output = [];
  if (Array.isArray(message.tool_calls) && message.tool_calls.length) {
    for (const call of message.tool_calls) {
      output.push({
        id: `fc_${crypto.randomUUID().replaceAll('-', '')}`,
        type: 'function_call',
        status: 'completed',
        call_id: call.id,
        name: call.function?.name,
        arguments: call.function?.arguments || '{}',
      });
    }
  } else {
    output.push({
      id: `msg_${crypto.randomUUID().replaceAll('-', '')}`,
      type: 'message', status: 'completed', role: 'assistant',
      content: [{ type: 'output_text', text: contentText(message.content), annotations: [] }],
    });
  }
  messages.push({
    role: 'assistant', content: contentText(message.content || ''),
    ...(Array.isArray(message.tool_calls) && message.tool_calls.length ? { tool_calls: message.tool_calls } : {}),
  });
  conversations.set(responseId, messages);
  const usage = chat.usage || {};
  return json(res, 200, {
    id: responseId, object: 'response', created_at: Math.floor(Date.now() / 1000),
    status: 'completed', model: chat.model || request.model, output,
    output_text: contentText(message.content || ''),
    parallel_tool_calls: Boolean(request.parallel_tool_calls),
    usage: {
      input_tokens: usage.prompt_tokens || 0,
      output_tokens: usage.completion_tokens || 0,
      total_tokens: usage.total_tokens || ((usage.prompt_tokens || 0) + (usage.completion_tokens || 0)),
      input_tokens_details: { cached_tokens: usage.prompt_tokens_details?.cached_tokens || 0 },
      output_tokens_details: { reasoning_tokens: usage.completion_tokens_details?.reasoning_tokens || 0 },
    },
  });
}

async function handleChat(req, res) {
  let body;
  try { body = await readJson(req); } catch { return json(res, 400, { error: { message: 'Invalid JSON request body.' } }); }
  try { return json(res, 200, await upstreamChat(body)); }
  catch (error) { return json(res, error.status || 502, error.payload || { error: { message: String(error.message || error) } }); }
}

http.createServer((req, res) => {
  const url = new URL(req.url || '/', 'http://localhost');
  if (req.method !== 'POST') return json(res, 404, { error: { message: 'Not found.' } });
  if (url.pathname === '/v1/responses') return void handleResponses(req, res);
  if (url.pathname === '/v1/chat/completions') return void handleChat(req, res);
  return json(res, 404, { error: { message: 'Not found.' } });
}).listen(port, '0.0.0.0', () => {
  console.error(`KSI Responses-to-Chat proxy listening on port ${port}.`);
});
