import { Outlet } from 'react-router-dom';

// Top-level layout shell. Children come from the router (`Landing` at
// `/`, `ConversationView` at `/c/:id`). We deliberately keep this
// minimal — header chrome lives inside each child so the conversation
// view can size its title bar freely.
export default function App() {
  return (
    <div className="app">
      <Outlet />
    </div>
  );
}
