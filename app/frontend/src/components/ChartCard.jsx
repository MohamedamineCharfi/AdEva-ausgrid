import React, { useEffect, useRef, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import { useTheme } from "../context/ThemeContext";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  zoomPlugin
);

const ChartCard = ({ title, data, fullWidth = false }) => {
  const { darkMode } = useTheme();
  const chartRef = useRef(null);

  const resetZoom = () => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
    }
  };

  const handleBlur = (e) => {
    // Only reset if focus leaves the component entirely
    if (!e.currentTarget.contains(e.relatedTarget)) {
      resetZoom();
    }
  };

  const wrapperRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        resetZoom();
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const chartData = {
    labels: data.labels,
    datasets: [
      {
        label: title,
        data: data.values,
        fill: true,
        backgroundColor: (context) => {
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, 0, 0, 300);
          gradient.addColorStop(
            0,
            darkMode ? "rgba(0, 120, 255, 0.4)" : "rgba(0, 120, 255, 0.3)"
          );
          gradient.addColorStop(
            1,
            darkMode ? "rgba(0, 120, 255, 0.05)" : "rgba(0, 120, 255, 0.02)"
          );
          return gradient;
        },
        borderColor: darkMode ? "#1E90FF" : "#0070f3",
        pointBackgroundColor: darkMode ? "#1E90FF" : "#0070f3",
        pointBorderColor: darkMode ? "#fff" : "#000",
        borderWidth: 2,
        pointRadius: title === "Daily Consumption" ? 0 : 3, // Make points smaller for daily chart
        pointHoverRadius: title === "Daily Consumption" ? 4 : 6, // Smaller hover radius for daily chart
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        enabled: true,
        mode: "index",
        intersect: false,
        callbacks: {
          label: (tooltipItem) =>
            `${tooltipItem.dataset.label}: ${tooltipItem.raw}`,
        },
        backgroundColor: "#222",
        titleColor: "#fff",
        bodyColor: "#fff",
        borderColor: "rgba(255,255,255,0.7)",
        borderWidth: 1,
      },
      zoom: {
        pan: {
          enabled: true,
          mode: "xy",
        },
        zoom: {
          wheel: { enabled: true },
          pinch: { enabled: true },
          mode: "xy",
        },
        limits: {
          x: { minRange: 1 },
          y: { minRange: 1 },
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: darkMode ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)",
        },
        ticks: {
          color: darkMode ? "#fff" : "#000",
          font: { size: 12 },
        },
      },
      y: {
        grid: {
          color: darkMode ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)",
        },
        ticks: {
          color: darkMode ? "#fff" : "#000",
          font: { size: 12 },
        },
      },
    },
  };

  // swap in the right classes based on `fullWidth`
  const wrapperClasses = fullWidth
    ? "rounded-xl shadow-lg p-6 w-full h-100 transition-colors duration-300 outline-none"
    : "rounded-xl shadow-lg p-4 flex flex-col items-center justify-center min-w-[220px] max-w-[350px] min-h-[260px] max-h-[320px] w-full h-full transition-colors duration-300 outline-none";

  const chartContainerClasses = fullWidth
    ? "w-full h-full"
    : "w-full h-48 md:h-40 lg:h-48 xl:h-56";
    
  return (
    <div
      tabIndex={0}
      ref={wrapperRef}
      onBlur={handleBlur}
      className={wrapperClasses}
      style={{
        backgroundColor: darkMode
          ? "rgba(30, 30, 30, 0.8)"
          : "rgba(245, 245, 245, 0.9)",
      }}
    >
      <h3
        className="text-lg font-semibold mb-2 text-center"
        style={{ color: darkMode ? "#fff" : "#000" }}
      >
        {title}
      </h3>
      <div className={chartContainerClasses}>
        <Line
          data={chartData}
          options={chartOptions}
          ref={(el) => {
            if (el) {
              chartRef.current = el.chartInstance || el.chart || el;
            }
          }}
        />
      </div>
    </div>
  );
};

export default ChartCard;
