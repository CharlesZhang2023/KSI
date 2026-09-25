import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const tsxSkip = fs.existsSync(tsxBin) ? undefined : 'tsx not installed';

describe('withThinkingPolicy', { skip: tsxSkip }, () => {
  it('adds thinking: disabled only when KSI_ANTHROPIC_DISABLE_THINKING is truthy', () => {
    const r = spawnSync(tsxBin, ['--input-type=module', '--eval', `
      import { withThinkingPolicy } from './agent-runner/src/anthropic_direct_transport.ts';
      console.log(JSON.stringify([
        withThinkingPolicy({}, { model: 'm' }),
        withThinkingPolicy({ KSI_ANTHROPIC_DISABLE_THINKING: '1' }, { model: 'm' }),
        withThinkingPolicy({ KSI_ANTHROPIC_DISABLE_THINKING: '1' }, { model: 'm', thinking: { type: 'enabled' } }),
      ]));
    `], { cwd: path.join(repoRoot, 'runtime_runner'), encoding: 'utf-8', env: { ...process.env, NODE_NO_WARNINGS: '1' } });
    assert.equal(r.status, 0, r.stderr);
    assert.deepEqual(JSON.parse(r.stdout.trim()), [
      { model: 'm' },
      { model: 'm', thinking: { type: 'disabled' } },
      { model: 'm', thinking: { type: 'enabled' } },
    ]);
  });
});
