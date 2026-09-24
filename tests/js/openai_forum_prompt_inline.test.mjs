/**
 * OpenAI forum phases are MCP-only, so the agent cannot read TASK.md, which
 * is the only place the host puts the forum instructions. inlineForumTaskMd
 * (forum_prompt.ts) appends the file contents to the initial prompt.
 */

import { strict as assert } from 'node:assert';
import { spawnSync } from 'node:child_process';
import { describe, it } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tsxBin = path.join(repoRoot, 'runtime_runner', 'node_modules', '.bin', 'tsx');
const tsxSkip = fs.existsSync(tsxBin) ? undefined : 'runtime_runner/node_modules/.bin/tsx is not installed';

function inline(prompt, taskMdPath) {
  const source = `
    import { inlineForumTaskMd } from './agent-runner/src/forum_prompt.ts';
    console.log(JSON.stringify(inlineForumTaskMd(${JSON.stringify(prompt)}, ${JSON.stringify(taskMdPath)})));
  `;
  const result = spawnSync(tsxBin, ['--input-type=module', '--eval', source], {
    cwd: path.join(repoRoot, 'runtime_runner'),
    encoding: 'utf-8',
    env: { ...process.env, NODE_NO_WARNINGS: '1' },
  });
  assert.equal(result.status, 0, result.stderr);
  return JSON.parse(result.stdout.trim());
}

describe('inlineForumTaskMd', { skip: tsxSkip }, () => {
  it('appends the forum TASK.md contents to the prompt', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-forum-inline-'));
    try {
      const taskMd = path.join(dir, 'TASK.md');
      fs.writeFileSync(taskMd, '# PER-TASK POST-MORTEM\nCall forum_post.\n');
      const out = inline('Read TASK.md.', taskMd);
      assert.ok(out.startsWith('Read TASK.md.'));
      assert.match(out, /<TASK\.md>\n# PER-TASK POST-MORTEM\nCall forum_post\.\n<\/TASK\.md>/);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('returns the prompt unchanged when TASK.md is missing or empty', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ksi-forum-inline-'));
    try {
      assert.equal(inline('P', path.join(dir, 'missing.md')), 'P');
      const empty = path.join(dir, 'TASK.md');
      fs.writeFileSync(empty, '   \n');
      assert.equal(inline('P', empty), 'P');
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
