import React, { useState } from "react";
import { UserGroupIcon, MapPinIcon } from "@heroicons/react/24/outline";
import { useTheme } from "../context/ThemeContext";
import Select from "react-select";

const FilterConsumers = ({ onApply }) => {
  const [consumerId, setConsumerId] = useState("");
  const [postcode, setPostcode] = useState("");
  const { darkMode } = useTheme();

  const handleApply = () => {
    // Trim whitespace and pass only non-empty values
    const filters = {
      consumerId: consumerId.trim() || undefined,
      postcode: postcode.trim() || undefined,
    };
    onApply(filters);
  };

  return (
    <div
      className={`flex flex-col rounded-xl shadow p-4 min-h-[110px] justify-center items-center mx-auto ${
        darkMode ? "bg-gray-800" : "bg-white"
      } ${darkMode ? "text-white" : "text-black"} md:ml-64`}
    >
      <h2
        className={`text-lg font-medium mb-4 ${darkMode ? "text-white" : ""}`}
      >
        Filter Consumers
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Consumer ID Input */}
        <div className="flex flex-col">
          <label
            className={`text-sm font-medium mb-1 ${
              darkMode ? "text-gray-300" : "text-gray-500"
            }`}
          >
            Consumer ID
          </label>
          <div className="flex items-center gap-4">
            <span className="rounded-lg p-3 bg-blue-100 flex items-center justify-center h-12 w-12">
              <UserGroupIcon className="h-6 w-6 text-blue-600" />
            </span>
            <Select
              options={["001", "002", "003", "010", "100"]
                .sort()
                .map((id) => ({ label: id, value: id }))}
              onChange={(selected) => setConsumerId(selected?.value || "")}
              placeholder="Select or type Consumer ID"
              className="w-full"
              isClearable
              styles={{
                control: (base) => ({
                  ...base,
                  backgroundColor: darkMode ? "#374151" : "white",
                  borderColor: darkMode ? "#4B5563" : "#D1D5DB",
                  color: darkMode ? "white" : "black",
                }),
                singleValue: (base) => ({
                  ...base,
                  color: darkMode ? "white" : "black",
                }),
                input: (base) => ({
                  ...base,
                  color: darkMode ? "white" : "black",
                }),
              }}
            />
          </div>
        </div>

        {/* Postcode Input */}
        <div className="flex flex-col">
          <label
            className={`text-sm font-medium mb-1 ${
              darkMode ? "text-gray-300" : "text-gray-500"
            }`}
          >
            Postcode
          </label>
          <div className="flex items-center gap-4">
            <span className="rounded-lg p-3 bg-green-100 flex items-center justify-center h-12 w-12">
              <MapPinIcon className="h-6 w-6 text-green-600" />
            </span>
            <input
              type="text"
              value={postcode}
              onChange={(e) => setPostcode(e.target.value)}
              placeholder="Enter Postcode"
              className={`mt-1 border rounded-lg px-4 py-2 w-full focus:ring ${
                darkMode ? "bg-gray-700 text-white border-gray-600" : ""
              }`}
            />
          </div>
        </div>
      </div>

      {/* Apply Filters Button */}
      <div className="mt-4 flex justify-end">
        <button
          onClick={handleApply}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg transition-all duration-200 hover:shadow-[0_0_12px_4px_rgba(59,130,246,0.7)] focus:outline-none"
        >
          APPLY FILTERS
        </button>
      </div>
    </div>
  );
};

export default FilterConsumers;
