import fs from 'fs';

/**
 * Inline the forum TASK.md into the initial prompt for OpenAI forum phases.
 *
 * The host puts the per-task / cross-task forum instructions only in the
 * workspace TASK.md, and the generic execution prompt says "Read TASK.md".
 * OpenAI forum phases are MCP-only (no shell / file tools, issue #1221), so
 * the agent cannot read that file and never sees the forum instructions. The
 * Anthropic direct forum adapter already receives the forum text directly;
 * this gives the OpenAI path the same content without widening its tool
 * surface (the runner process reads the file, not the agent).
 */
export function inlineForumTaskMd(prompt: string, taskMdPath: string): string {
  let taskMd = '';
  try {
    taskMd = fs.readFileSync(taskMdPath, 'utf-8').trim();
  } catch {
    return prompt;
  }
  if (!taskMd) return prompt;
  return [
    prompt.trimEnd(),
    '',
    'The forum phase has no file-reading tools, so the contents of TASK.md are provided below.',
    'Follow them using the forum tools (forum_read, knowledge, query, forum_post, forum_signal_done).',
    '',
    '<TASK.md>',
    taskMd,
    '</TASK.md>',
    '',
  ].join('\n');
}
