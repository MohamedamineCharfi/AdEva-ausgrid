import {
  HomeModernIcon,
  UserGroupIcon,
  BellIcon,
  CogIcon,
  ArrowRightEndOnRectangleIcon,
} from "@heroicons/react/24/outline";
import { useNavigate } from "react-router-dom";
import { logout } from "../services/auth";
import React, { useState } from "react";
import { useTheme } from "../context/ThemeContext";

export default function Sidebar() {
  const navigate = useNavigate();
  const [active, setActive] = useState("Dashboard");
  const { darkMode } = useTheme();

  const handleLogout = () => {
    logout();
  };

  const menuItems = [
    { name: "Dashboard", icon: HomeModernIcon, route: "/" },
    { name: "Consumers List", icon: UserGroupIcon, route: "/consumers" },
    { name: "Alerts", icon: BellIcon },
    { name: "Profile", icon: CogIcon },
  ];

  return (
    <aside
      className={`fixed left-0 top-2.5 h-[95vh] w-64 flex flex-col rounded-xl m-2 mb-8 shadow-2xl transition-colors duration-300 ${
        darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-800'
      }`}
    >
      {/* Header Section */}
      <div className="px-6 py-8 flex items-center gap-3">
        <span className="rounded-lg p-1.5 bg-transparent">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M13 3L5 15.5H13L11 25L23 10.5H15L17 3Z" fill="#FF7A1A"/>
          </svg>
        </span>
        <span
          className={`text-2xl font-extrabold tracking-tight transition-colors duration-300 ${darkMode ? 'text-white' : 'text-black'}`}
          style={{ letterSpacing: '-0.02em' }}
        >
          Ausgrid
        </span>
      </div>

      {/* Menu Items */}
      <nav className="flex-1 px-4 space-y-1 overflow-y-auto">
        {menuItems.map((item) => (
          <button
            key={item.name}
            className="group flex items-center px-2 py-2 text-base font-medium rounded-md hover:bg-blue-100 hover:text-blue-700 w-full text-left"
            onClick={() => navigate(item.route)}
          >
            <item.icon className="mr-4 h-6 w-6 text-blue-400 group-hover:text-blue-700" />
            {item.name}
          </button>
        ))}
      </nav>

      {/* Logout Button */}
      <div className="px-4 pb-4 mt-2">
        <button
          onClick={handleLogout}
          className="flex items-center w-full px-4 py-3 rounded-lg bg-red-600 text-white hover:bg-red-700 transition-all mb-4"
        >
          <ArrowRightEndOnRectangleIcon className="h-6 w-6 mr-4" />
          <span className="text-base">Logout</span>
        </button>
      </div>
    </aside>
  );
}