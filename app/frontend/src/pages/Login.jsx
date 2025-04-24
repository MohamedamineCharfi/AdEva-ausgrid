import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../services/auth";
import { Zap, BatteryCharging, PlugZap, Sun } from "lucide-react";
import bgImage from "../assets/login-bg.png"; // ← rename your file to login-bg.png

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();
  

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      await login(username, password);
      navigate("/");
    } catch (error) {
      const msg = error.response?.data?.detail || error.message;
      alert("Login failed: " + msg);
    }
  };

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Blurred background image */}
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{
          backgroundImage: `url(${bgImage})`,
          filter: "blur(8px)",
          opacity: 0.7,
        }}
      />

      {/* Dark overlay */}
      <div className="absolute inset-0 bg-black opacity-30" />

      {/* Centered form */}
      <div className="relative z-10 flex items-center justify-center h-full px-4">
        <form
          onSubmit={handleLogin}
          className="w-full max-w-sm bg-white bg-opacity-90 p-8 rounded-2xl shadow-2xl backdrop-blur-sm"
        >
          <h2 className="text-3xl font-bold mb-6 text-center text-blue-600">
            Energy Manager
          </h2>
          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full p-2 mb-4 border rounded focus:outline-none focus:ring-2 focus:ring-blue-400"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-2 mb-6 border rounded focus:outline-none focus:ring-2 focus:ring-blue-400"
            required
          />
          <button
            type="submit"
            className="w-full bg-blue-600 text-white p-2 rounded transition duration-200 hover:bg-blue-700 focus:shadow-outline"
          >
            Login
          </button>
          <p className="mt-4 text-center text-sm">
            Don’t have an account?{" "}
            <a href="/register" className="text-blue-500 hover:underline">
              Register here
            </a>
          </p>
        </form>
      </div>
    </div>
  );
}