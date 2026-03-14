import axios from 'axios';

// Base API URL — override with VITE_API_URL environment variable at build time.
// In Docker, set VITE_API_URL='' so all /api/* calls go through the Nginx proxy.
// For local dev (npm run dev), leave VITE_API_URL unset/empty — the Vite dev-server
// proxy (vite.config.js) will forward all /api/* etc. to http://localhost:8000.
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// Axios instance with default configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

const CACHE = {};
const CACHE_TTL = 300000; // 5 minutes cache (300k ms) to prevent loading flickers on tab switches
export const getCachedData = (url) => {
  const now = Date.now();
  if (CACHE[url] && (now - CACHE[url].timestamp < CACHE_TTL)) {
    return CACHE[url].data;
  }
  return null;
};
const cachedGet = async (url) => {
  const now = Date.now();
  if (CACHE[url] && (now - CACHE[url].timestamp < CACHE_TTL)) {
    return CACHE[url].data;
  }
  const response = await api.get(url);
  CACHE[url] = { data: response.data, timestamp: now };
  return response.data;
};

// Services for API endpoints
export const AnalystService = {
  runPipeline: async (datasetId) => {
    try {
      const response = await api.post('/analyst/run', { dataset_id: datasetId });
      return response.data;
    } catch (error) {
      console.error('Error running pipeline:', error);
      throw error;
    }
  },

  getSystemMetrics: async () => {
    try {
      const response = await api.get('/metrics');
      return response.data;
    } catch (error) {
      console.error('Error fetching system metrics:', error);
      throw error;
    }
  }
};

export const ResultsService = {
  getAllResults: async () => {
    try {
      const response = await cachedGet('/api/results');
      return response;
    } catch (error) {
      console.error('Error fetching all results:', error);
      return { runs: [], total: 0 };
    }
  },

  getLatestResult: async () => {
    try {
      const response = await cachedGet('/api/results/latest');
      return response;
    } catch (error) {
      console.error('Error fetching latest result:', error);
      if (error.response?.status === 404) {
        return null;
      }
      throw error;
    }
  },

  getResult: async (runId) => {
    try {
      const response = await cachedGet(`/api/results/${runId}`);
      return response;
    } catch (error) {
      console.error(`Error fetching result for run ${runId}:`, error);
      throw error;
    }
  },

  getIntelligence: async (runId) => {
    try {
      const response = await cachedGet(`/api/results/${runId}/intelligence`);
      return response;
    } catch (error) {
      console.error(`Error fetching intelligence for run ${runId}:`, error);
      return null;
    }
  }
};

export const CohortService = {
  getRetention: async () => {
    try {
      const response = await api.post('/api/cohort', {});
      return response.data;
    } catch (error) {
      console.error('Error fetching cohort retention:', error);
      throw error;
    }
  }
};

export default api;
