import axios from "axios";

export const API = axios.create({
  baseURL: "http://localhost:8000/api",
});

API.interceptors.request.use(async (config) => {
  let token = localStorage.getItem("access_token");

  // If expired, try refreshing it
  if (token) {
    const { exp } = JSON.parse(atob(token.split(".")[1]));
    if (exp * 1000 < Date.now()) {
      try {
        const res = await axios.post("/api/token/refresh/", {
          refresh: localStorage.getItem("refresh_token"),
        });
        localStorage.setItem("access_token", res.data.access);
        token = res.data.access;
      } catch (err) {
        console.error("Refresh token failed", err);
        localStorage.clear();
        window.location.href = "/login"; // Redirect to login
      }
    }
    console.log("Access Token:", localStorage.getItem("access_token"));
    config.headers.Authorization = `Bearer ${token}`;
  }

  return config;
});
// Fetch all consumers
export const fetchConsumers = (filters = {}) =>
  API.get("/consumers/", {
    params: filters, // Pass filters as query parameters
    headers: { Authorization: `Bearer ${localStorage.getItem("access_token")}` },
  }).then((res) => res.data);

// Fetch records (optionally filtered)  
export const fetchRecords = ({ consumerId, postcode, startDate, endDate } = {}) => {
  const params = {};
  if (consumerId && consumerId !== "" && consumerId !== "all") params.Customer   = consumerId;
  if (postcode && postcode !== "" && postcode !== "all")    params.Postcode   = postcode;
  if (startDate)   params.start_date = startDate;
  if (endDate)     params.end_date   = endDate;

  return API.get("/records/", {
    headers: { Authorization: `Bearer ${localStorage.getItem("access_token")}` },
    params,
  }).then((res) => res.data);
};

export const predictConsumption = async (records) => {
  try {
    const response = await fetch("/api/predict/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ records }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data?.error || "Prediction failed.");
    }

    return data.predictions;
  } catch (error) {
    throw new Error(error.message || "Server error during prediction.");
  }
};
