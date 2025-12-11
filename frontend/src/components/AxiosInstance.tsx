import axios from "axios";

// const baseUrl = "http://127.0.0.1:8000/";
// const rawApiBase = process.env.REACT_APP_API_URL || '';
// Access the environment variable using import.meta.env
const rawApiBase = import.meta.env.VITE_REACT_APP_API_URL || ''; 
console.log("rawApiBase: ", rawApiBase);

console.log("All ENV variables:", import.meta.env);

const apiBase = rawApiBase.replace(/\/+$/,''); // remove trailing slash if present
const baseUrl = apiBase ? `${apiBase}` : '/api';

const AxiosInstance = axios.create({
  baseURL: baseUrl,
  timeout: 300000,
  headers: {
    accept: "application/json",
  },
});

AxiosInstance.interceptors.request.use((config) => {
  const token = localStorage.getItem("Token");
  if (token) {
    config.headers.Authorization = `Token ${token}`;
  } else {
    config.headers.Authorization = ``;
  }
  return config;
});

AxiosInstance.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem("Token");
    }

    return Promise.reject(error);
  }
);

export default AxiosInstance;
