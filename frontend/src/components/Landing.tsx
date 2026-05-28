import { usePageTitle } from '../title';

export default function Landing() {
  usePageTitle('Viewer');
  return (
    <main className="center">
      <h1>grok viewer</h1>
      <p>
        Conversation traces are accessed by their unguessable UUID,
        surfaced as a link in Discord when the bot opens a new
        conversation.
      </p>
    </main>
  );
}
