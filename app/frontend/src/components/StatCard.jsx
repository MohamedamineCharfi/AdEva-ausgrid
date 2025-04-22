import { useTheme } from "../context/ThemeContext";

export default function StatCard({ icon, label, value }) {
  const { darkMode } = useTheme();

  return (
    <div
      className={`rounded-xl shadow p-4 flex items-center space-x-4 ${
        darkMode ? "bg-gray-800" : "bg-white"
      }`}
    >
      {/* Icon container */}
      <div
        className={`p-3 rounded-lg ${
          darkMode ? "bg-gray-700" : "bg-gray-100"
        } flex items-center justify-center`}
      >
        {icon}
      </div>

      {/* Label and Value */}
      <div>
        <div
          className={`text-sm font-medium ${
            darkMode ? "text-gray-300" : "text-gray-500"
          }`}
        >
          {label}
        </div>
        <div
          className={`text-2xl font-semibold ${
            darkMode ? "text-white" : "text-black"
          }`}
        >
          {value}
        </div>
      </div>
    </div>
  );
}