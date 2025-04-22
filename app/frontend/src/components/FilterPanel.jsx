import React, { useState } from "react";
import Select from "react-select";
import {
  UserGroupIcon,
  MapPinIcon,
  CalendarIcon,
} from "@heroicons/react/24/outline";
import { useTheme } from "../context/ThemeContext";

export default function FilterPanel({
  consumers,
  postcodes,
  onApply,
  onReset,
}) {
  const { darkMode } = useTheme();

  // controlled states
  const [consumerId, setConsumerId] = useState("");
  const [postcode, setPostcode] = useState("");
  const [consumerInput, setConsumerInput] = useState("");
  const [postcodeInput, setPostcodeInput] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  // build & sort options
  const consumerOptions = [
    { value: "", label: "All Consumers" },
    ...consumers
      .map((c) => c.Customer)
      .sort((a, b) => a - b)
      .map((id) => ({ value: String(id), label: `Consumer ${id}` })),
  ];

  const postcodeOptions = [
    { value: "", label: "All Postcodes" },
    ...consumers
      .map((c) => c.Postcode)
      .sort((a, b) => a - b)
      .map((id) => ({ value: String(id), label: `Postcode ${id}` })),
  ];

  // handlers clear both value + input text
  const handleConsumerChange = (sel) => {
    setConsumerId(sel?.value || "");
    setConsumerInput("");
  };
  const handlePostcodeChange = (sel) => {
    setPostcode(sel?.value || "");
    setPostcodeInput("");
  };

  const handleApply = () =>
    onApply({ consumerId, postcode, startDate, endDate });
  const handleReset = () => {
    setConsumerId("");
    setPostcode("");
    setConsumerInput("");
    setPostcodeInput("");
    setStartDate("");
    setEndDate("");
    onReset();
  };

  // common react‑select style overrides for dark/light
  const rsStyles = {
    control: (base) => ({
      ...base,
      backgroundColor: darkMode ? "#374151" : "#fff",
      borderColor: darkMode ? "#4B5563" : "#D1D5DB",
    }),
    singleValue: (base) => ({
      ...base,
      color: darkMode ? "#fff" : "#000",
    }),
    input: (base) => ({
      ...base,
      color: darkMode ? "#fff" : "#000",
    }),
    menu: (base) => ({
      ...base,
      backgroundColor: darkMode ? "#374151" : "#fff",
    }),
    option: (base, state) => ({
      ...base,
      backgroundColor: state.isFocused
        ? darkMode
          ? "#4B5563"
          : "#E5E7EB"
        : "transparent",
      color: darkMode ? "#fff" : "#000",
    }),
  };

  return (
    <div
      className={`rounded-xl shadow p-6 mb-6 ${
        darkMode ? "bg-gray-800 text-white" : "bg-white"
      }`}
    >
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
        {/* Consumer Select */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Consumer
          </label>
          <Select
            options={consumerOptions}
            isClearable
            placeholder="Select or type…"
            value={
              consumerId
                ? consumerOptions.find((o) => o.value === consumerId)
                : null
            }
            onChange={handleConsumerChange}
            inputValue={consumerInput}
            onInputChange={setConsumerInput}
            styles={rsStyles}
            components={{ DropdownIndicator: null }}
          />
        </div>

        {/* Postcode Select */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Postcode
          </label>
          <Select
            options={postcodeOptions}
            isClearable
            placeholder="Select or type…"
            value={
              postcode
                ? postcodeOptions.find((o) => o.value === postcode)
                : null
            }
            onChange={handlePostcodeChange}
            inputValue={postcodeInput}
            onInputChange={setPostcodeInput}
            styles={rsStyles}
            components={{ DropdownIndicator: null }}
          />
        </div>

        {/* Start Date */}
        <div>
          <label className="block text-sm font-medium mb-1">
            Start Date
          </label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="border rounded-lg px-4 py-2 w-full focus:ring"
          />
        </div>

        {/* End Date */}
        <div>
          <label className="block text-sm font-medium mb-1">
            End Date
          </label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="border rounded-lg px-4 py-2 w-full focus:ring"
          />
        </div>
      </div>

      <div className="flex justify-end space-x-2">
        <button
          onClick={handleReset}
          className="px-4 py-2 border rounded-lg hover:bg-gray-100"
        >
          RESET
        </button>
        <button
          onClick={handleApply}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          APPLY FILTERS
        </button>
      </div>
    </div>
  );
}
