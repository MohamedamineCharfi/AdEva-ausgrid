import { MagnifyingGlassIcon, UserCircleIcon, CogIcon, BellIcon } from "@heroicons/react/24/outline";
import ThemeToggle from "./ThemeToggle";
import { useTheme } from "../context/ThemeContext";

export default function Header() {
  const { darkMode } = useTheme();
  
  return (
    <header className={`flex items-center justify-between ${darkMode ? 'bg-dark-bg text-dark-text' : 'bg-gray-100'} px-6 py-4 transition-colors duration-300`}>
      <div>
        <nav className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          <a href="#" className={`${darkMode ? 'hover:text-gray-200' : 'hover:text-gray-700'}`}>Dashboard</a>
        </nav>
        <h1 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>Dashboard</h1>
      </div>
      <div className="flex items-center space-x-4">
        <ThemeToggle />
        <div className="relative">
          <MagnifyingGlassIcon className={`h-5 w-5 ${darkMode ? 'text-gray-500' : 'text-gray-400'} absolute left-3 top-1/2 transform -translate-y-1/2`} />
          <input
            type="text"
            placeholder="Search here"
            className={`pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring ${darkMode ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-500' : ''}`}
          />
        </div>
        <UserCircleIcon className={`h-6 w-6 ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-800'} cursor-pointer`} />
        <CogIcon className={`h-6 w-6 ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-800'} cursor-pointer`} />
        <BellIcon className={`h-6 w-6 ${darkMode ? 'text-gray-400 hover:text-white' : 'text-gray-600 hover:text-gray-800'} cursor-pointer`} />
      </div>
    </header>
  );
}
