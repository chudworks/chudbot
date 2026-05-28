import { useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import { useSiteConfig } from './title';

// Top-level layout shell. Children come from the router (`Landing` at
// `/`, `ConversationView` at `/c/:id`). We deliberately keep this
// minimal — header chrome lives inside each child so the conversation
// view can size its title bar freely.
export default function App() {
  // Pull the server-configured tab-title prefix once, up front. Every
  // page's `usePageTitle` reads it from the same store.
  const loadConfig = useSiteConfig((s) => s.load);
  useEffect(() => {
    void loadConfig();
  }, [loadConfig]);

  return (
    <div className="app">
      <Outlet />
    </div>
  );
}
