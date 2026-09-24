/**
 * GatewayFailoverModel (gateway_failover_model.ts): on a gateway-failover
 * 400 about encrypted reasoning state, re-send the same turn once without
 * replayed reasoning items and item ids; leave every other path untouched.
 */

import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const tsxSkip = fs.existsSync(tsxBin) ? undefined : 'runtime_runner/node_modules/.bin/tsx is not installed';

function runFixture(body) {
  const source = `
    import { GatewayFailoverModel, isGatewayStateError, stripGatewayState } from './agent-runner/src/gateway_failover_model.ts';
    ${body}
  `;
  const result = spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    cwd: path.join(repoRoot, 'runtime_runner'),
    encoding: 'utf-8',
    env: { ...process.env, NODE_NO_WARNINGS: '1' },
  });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout.trim());
}

const INPUT = `[
  { role: 'user', content: 'solve' },
  { type: 'reasoning', id: 'rs_1', content: [], providerData: { encrypted_content: 'x' } },
  { type: 'function_call', id: 'fc_1', callId: 'c1', name: 'add', arguments: '{}' },
  { type: 'function_call_result', callId: 'c1', output: '3' },
]`;

describe('GatewayFailoverModel', { skip: tsxSkip }, () => {
  it('re-sends once without reasoning items or ids on a gateway state error', () => {
    const out = runFixture(`
      const calls = [];
      const inner = {
        async getResponse(req) {
          calls.push(req.input);
          if (calls.length === 1) throw new Error('400 The encrypted content for item *** could not be verified.');
          return { output: [], usage: {} };
        },
        getStreamedResponse() { throw new Error('unused'); },
      };
      const model = new GatewayFailoverModel(inner);
      await model.getResponse({ input: ${INPUT}, previousResponseId: undefined });
      console.log(JSON.stringify({ calls, recoveries: model.recoveries }));
    `);
    assert.equal(out.calls.length, 2);
    assert.equal(out.recoveries, 1);
    assert.equal(out.calls[0].length, 4);
    const retried = out.calls[1];
    assert.equal(retried.length, 3);
    assert.ok(retried.every((item) => item.type !== 'reasoning' && !('id' in item)));
    assert.equal(retried[1].callId, 'c1');
  });

  it('rethrows unrelated errors and a second gateway error', () => {
    const out = runFixture(`
      const run = async (messages) => {
        let n = 0;
        const inner = {
          async getResponse() { throw new Error(messages[Math.min(n++, messages.length - 1)]); },
          getStreamedResponse() { throw new Error('unused'); },
        };
        try { await new GatewayFailoverModel(inner).getResponse({ input: ${INPUT} }); return 'ok'; }
        catch (e) { return String(e.message).slice(0, 20) + '|' + n; }
      };
      console.log(JSON.stringify([
        await run(['429 rate limit']),
        await run(['400 The conversation context is not compatible with the available resources', '400 The conversation context is not compatible with the available resources']),
      ]));
    `);
    assert.deepEqual(out, ['429 rate limit|1', '400 The conversation|2']);
  });

  it('classifies only the gateway state messages', () => {
    const out = runFixture(`
      console.log(JSON.stringify([
        isGatewayStateError(new Error('400 The encrypted conversation context could not be validated.')),
        isGatewayStateError(new Error('400 invalid prompt')),
        stripGatewayState('plain string input'),
      ]));
    `);
    assert.deepEqual(out, [true, false, 'plain string input']);
  });
});
