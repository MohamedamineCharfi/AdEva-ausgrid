import React from 'react';
import { ExclamationCircleIcon } from '@heroicons/react/24/outline'; // Corrected import
import { useTheme } from '../context/ThemeContext';

const Notification = ({ message, onClose }) => {
  const { darkMode } = useTheme();

  return (
    <div className={`fixed bottom-4 left-4 flex items-center gap-4 p-4 rounded-lg shadow-lg ${darkMode ? 'bg-red-800 text-white' : 'bg-red-100 text-red-800'}`}>
      <ExclamationCircleIcon className="h-6 w-6" /> {/* Corrected icon */}
      <div className="flex-1">
        <span className="font-medium">Material Dashboard</span>
        <p>{message}</p>
      </div>
      <button onClick={onClose} className="text-sm font-medium hover:underline">
        Close
      </button>
    </div>
  );
};

export default Notification;