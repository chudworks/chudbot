import { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { useSiteConfig } from './title';

// Top-level layout shell. Children come from the router (`Landing` at
// `/`, `ConversationView` at `/c/:id`). We deliberately keep this
// minimal — header chrome lives inside each child so the conversation
// view can size its title bar freely. The version footer lives here so
// it shows on every page.
export default function App() {
  // Pull the server config once, up front. Every page's `usePageTitle`
  // reads the title prefix from the same store; the footer reads the
  // build version from it.
  const loadConfig = useSiteConfig((s) => s.load);
  const version = useSiteConfig((s) => s.version);
  useEffect(() => {
    void loadConfig();
  }, [loadConfig]);

  return (
    <div className="app">
      <Outlet />
      {version != null && (
        <footer className="app-footer">
          <span>{version}</span>
        </footer>
      )}
    </div>
  );
}
