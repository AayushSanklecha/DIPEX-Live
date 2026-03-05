import axios from 'axios';

// Base API URL configuration
const API_BASE_URL = 'http://localhost:8000';

// Axios instance with default configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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
  getLatestResult: async () => {
    try {
      const response = await api.get('/api/results/latest');
      return response.data;
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
      const response = await api.get(`/api/results/${runId}`);
      return response.data;
    } catch (error) {
      console.error(`Error fetching result for run ${runId}:`, error);
      throw error;
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
