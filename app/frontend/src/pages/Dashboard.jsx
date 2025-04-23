import React, { useEffect, useState } from "react";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import FilterPanel from "../components/FilterPanel";
import StatCard from "../components/StatCard";
import ChartCard from "../components/ChartCard";
import { useNavigate } from "react-router-dom";
import { fetchConsumers, fetchRecords } from "../services/api";
import { useTheme } from "../context/ThemeContext";
import {
  UsersIcon,
  MapPinIcon,
  CalendarIcon,
} from "@heroicons/react/24/outline";
import Notification from "../components/Notification";
import { ExclamationCircleIcon } from "@heroicons/react/24/outline";
import { ChartBarIcon } from "@heroicons/react/24/solid";
import { handlePredictionLogic } from "../services/predictHelper";

export default function Dashboard() {
  const [predictionData, setPredictionData] = useState(null);
  const [consumers, setConsumers] = useState([]);
  const [postcodes, setPostcodes] = useState([]);
  const [records, setRecords] = useState([]);
  const [stats, setStats] = useState({
    consumerId: "All",
    postcode: "All",
    startDate: "All",
    endDate: "All",
  });
  const navigate = useNavigate();

  useEffect(() => {
    if (!localStorage.getItem("access_token")) {
      navigate("/login");
    }
  }, [navigate]);

  // Load consumers once
  useEffect(() => {
    fetchConsumers().then(setConsumers);
  }, []);

  // Update postcodes when records change
  useEffect(() => {
    const pcs = [...new Set(records.map((r) => r.Postcode))];
    setPostcodes(pcs);
    console.log("Postcodes:", postcodes);
  }, [records]);

  // State for notifications
  const [notification, setNotification] = useState(null);

  // Apply filters
  const handleApply = ({ consumerId, postcode, startDate, endDate }) => {
    setPredictionData(null); // 🧼 Clear prediction when filters are applied
    console.log("Applying filters:", {
      consumerId,
      postcode,
      startDate,
      endDate,
    });
    fetchRecords({ consumerId, postcode, startDate, endDate }).then((data) => {
      if (data.length === 0) {
        setNotification("The consumer ID or postcode does not exist.");
      } else {
        setRecords(data);
        setStats({
          consumerId: consumerId || "All",
          postcode: postcode || "All",
          startDate: startDate || "All",
          endDate: endDate || "All",
        });
        setNotification(null);
      }
    });
  };
  

  // Reset filters
  const handleReset = () => {
    setPredictionData(null); // 🧼 Clear prediction on reset
    setRecords([]);
    setStats({
      consumerId: "All",
      postcode: "All",
      startDate: "All",
      endDate: "All",
    });
    setPostcodes([]);
    fetchRecords({}).then((data) => {
      setRecords(data);
      setStats({
        consumerId: "All",
        postcode: "All",
        startDate: "All",
        endDate: "All",
      });
    });
  };
  

  const handlePredict = () => {
    handlePredictionLogic({
      records,
      consumerId: stats.consumerId,
      setNotification,
      setPredictionData,
    });
  };

  // Helper function to aggregate data for charts
  const aggregateData = (records, period) => {
    const result = { labels: [], values: [] };

    records.forEach((record) => {
      let label = "";
      const date = new Date(record.date);

      if (period === "year") {
        label = date.getFullYear();
      } else if (period === "month") {
        label = `${date.getMonth() + 1}-${date.getFullYear()}`;
      } else if (period === "day") {
        label = date.toLocaleDateString();
      }

      if (!result.labels.includes(label)) {
        result.labels.push(label);
        result.values.push(record.consumption);
      } else {
        const index = result.labels.indexOf(label);
        result.values[index] += record.consumption;
      }
    });

    return result;
  };

  const yearlyData = aggregateData(records, "year");
  const monthlyData = aggregateData(records, "month");
  const dailyData = aggregateData(records, "day");

  // Get theme context
  const { darkMode } = useTheme();

  // Apply dark mode styling to the entire page (including body background)
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add("bg-dark-bg", "text-dark-text");
      document.body.classList.remove("bg-white", "text-gray-900");
    } else {
      document.body.classList.add("bg-white", "text-gray-900");
      document.body.classList.remove("bg-dark-bg", "text-dark-text");
    }
  }, [darkMode]);

  return (
    <div className="min-h-screen flex">
      <div
        className={`pl-64 flex flex-col w-full bg-gray-100 transition-colors duration-300 dark:bg-gray-900`}
      >
        <Sidebar />
        <div className="flex-1 flex flex-col">
          <Header />

          {/* Notification */}
          {notification && (
            <Notification
              message={notification}
              onClose={() => setNotification(null)}
              icon={<ExclamationCircleIcon className="h-6 w-6 text-red-600" />}
            />
          )}

          <main className="p-6 overflow-auto">
            <FilterPanel
              consumers={consumers}
              postcodes={postcodes}
              onApply={handleApply}
              onReset={handleReset}
              onPredict={handlePredict}
            />

            {/* Stats */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <StatCard
                icon={<UsersIcon className="h-6 w-6 text-blue-600" />}
                label="Consumer ID"
                value={stats.consumerId}
              />
              <StatCard
                icon={<MapPinIcon className="h-6 w-6 text-blue-600" />}
                label="Postcode"
                value={stats.postcode}
              />
              <StatCard
                icon={<CalendarIcon className="h-6 w-6 text-blue-600" />}
                label="Start Date"
                value={stats.startDate}
              />
              <StatCard
                icon={<CalendarIcon className="h-6 w-6 text-blue-600" />}
                label="End Date"
                value={stats.endDate}
              />
            </div>
            {/* Charts */}

            {/* Prediction chart — full width */}
            {predictionData && (
              <div className="mb-6">
                <ChartCard
                  title="Predicted Weekly Consumption"
                  data={predictionData}
                  fullWidth      
                />
              </div>
            )}

            {/* The rest of your charts in a 3-col grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-stretch">
              <ChartCard title="Daily Consumption" data={dailyData} />
              <ChartCard title="Monthly Consumption" data={monthlyData} />
              <ChartCard title="Yearly Consumption" data={yearlyData} />
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
