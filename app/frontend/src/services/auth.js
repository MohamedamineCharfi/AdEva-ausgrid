// services/auth.js
import { API } from "./api";

export async function login(username, password) {
  const { data } = await API.post("/login/", { username, password });
  localStorage.setItem("access_token",  data.access);
  localStorage.setItem("refresh_token", data.refresh);
  return data;
}

export function logout() {
  localStorage.clear();
  window.location.href = "/login";
}
