import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import App from './App';
import Landing from './components/Landing';
import ConversationView from './components/ConversationView';
import './styles/main.scss';

const root = document.getElementById('root');
if (!root) {
  throw new Error('No #root element found in index.html');
}

createRoot(root).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<Landing />} />
          <Route path="c/:id" element={<ConversationView />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>
);
