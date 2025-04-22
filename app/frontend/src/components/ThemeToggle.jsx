import React from 'react';
import { useTheme } from '../context/ThemeContext';
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';

const ThemeToggle = () => {
  const { darkMode, toggleTheme } = useTheme();

  return (
    <div className="flex items-center gap-2">
      <span className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-700'}`}>
        {darkMode ? 'Dark' : 'Light'}
      </span>
      <button
        onClick={toggleTheme}
        className="relative inline-flex h-8 w-16 items-center rounded-full transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        style={{
          backgroundColor: darkMode ? '#3b82f6' : '#d1d5db',
          boxShadow: '0 0 8px rgba(0, 0, 0, 0.2) inset'
        }}
        aria-label="Toggle dark mode"
      >
        <span
          className={`flex h-6 w-6 transform items-center justify-center rounded-full bg-white transition-transform duration-500 ${darkMode ? 'translate-x-9' : 'translate-x-1'}`}
        >
          {darkMode ? (
            <MoonIcon className="h-4 w-4 text-blue-600" />
          ) : (
            <SunIcon className="h-4 w-4 text-yellow-500" />
          )}
        </span>
        <span className="sr-only">{darkMode ? 'Dark Mode' : 'Light Mode'}</span>
      </button>
    </div>
  );
};

export default ThemeToggle;