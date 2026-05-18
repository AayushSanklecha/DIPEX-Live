import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Shell from './components/layout/Shell';
import RunPipeline from './pages/RunPipeline';
import ApiDocs from './pages/ApiDocs';
import Analytics from './pages/Analytics';
import { PipelineSessionProvider } from './context/PipelineSessionContext';

class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: '2rem', fontFamily: 'monospace', background: '#0a0a14', color: '#f87171', minHeight: '100vh' }}>
          <h2 style={{ color: '#fb7185', marginBottom: '1rem' }}>⚠️ Component Error</h2>
          <pre style={{ background: '#1a0a0a', padding: '1rem', borderRadius: '8px', border: '1px solid #7f1d1d', whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '0.82rem', color: '#fca5a5' }}>
            {this.state.error?.message}
            {'\n\n'}
            {this.state.error?.stack}
          </pre>
          <button onClick={() => this.setState({ error: null })}
            style={{ marginTop: '1rem', padding: '0.5rem 1rem', background: '#6366f1', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const App = () => {
  return (
    <PipelineSessionProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Shell />}>
            <Route index element={<ErrorBoundary><Analytics /></ErrorBoundary>} />
            <Route path="run" element={<ErrorBoundary><RunPipeline /></ErrorBoundary>} />
            <Route path="analytics" element={<Navigate to="/" replace />} />
            <Route path="docs" element={<ErrorBoundary><ApiDocs /></ErrorBoundary>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </PipelineSessionProvider>
  );
};

export default App;
