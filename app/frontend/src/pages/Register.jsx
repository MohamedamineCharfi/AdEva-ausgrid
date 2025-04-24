// src/pages/Register.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { API } from "../services/api";
import bgImage from "../assets/login-bg.png"; // same bg as Login

export default function Register() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleRegister = async (e) => {
    e.preventDefault();
    try {
      await API.post("/register/", { username, password });
      alert("Registration successful! Please login.");
      navigate("/login");
    } catch (error) {
      alert(
        "Registration failed: " +
          JSON.stringify(error.response?.data || error.message)
      );
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
          onSubmit={handleRegister}
          className="w-full max-w-sm bg-white bg-opacity-90 p-8 rounded-2xl shadow-2xl backdrop-blur-sm"
        >
          <h2 className="text-3xl font-bold mb-6 text-center text-green-600">
            Create Account
          </h2>

          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full p-2 mb-4 border rounded focus:outline-none focus:ring-2 focus:ring-green-400"
            required
          />

          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full p-2 mb-6 border rounded focus:outline-none focus:ring-2 focus:ring-green-400"
            required
          />

          <button
            type="submit"
            className="w-full bg-green-500 text-white p-2 rounded transition duration-200 hover:bg-green-600 focus:shadow-outline"
          >
            Register
          </button>

          <p className="mt-4 text-center text-sm">
            Already have an account?{" "}
            <a href="/login" className="text-green-500 hover:underline">
              Login here
            </a>
          </p>
        </form>
      </div>
    </div>
  );
}
