import React, { useEffect, useState } from "react";
import { fetchConsumers } from "../services/api";
import { Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography, CircularProgress, Box } from "@mui/material";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import FilterConsumers from "../components/FilterConsumers";
import { useTheme } from "../context/ThemeContext";

const ConsumerList = () => {
  const [consumers, setConsumers] = useState([]);
  const [filteredConsumers, setFilteredConsumers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const { darkMode } = useTheme();

  useEffect(() => {
    fetchConsumers()
      .then((data) => {
        console.log("Raw API Response:", data); // Debugging log
        setConsumers(data);
        setFilteredConsumers(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching consumers:", err); // Debugging log
        setError("Failed to load consumers");
        setLoading(false);
      });
  }, []);

  return (
    <div className={`pl-64 flex min-h-screen transition-colors duration-300 ${darkMode ? 'bg-dark-bg' : 'bg-gray-100'}`}>
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header />

        {/* Filter Panel */}
        <FilterConsumers
          onApply={(filters) => {
            console.log("Filters Applied:", filters); // Debugging log
            const filtered = consumers.filter((consumer) => {
              return (
                (!filters.consumerId || consumer.Customer?.toString().includes(filters.consumerId)) &&
                (!filters.postcode || consumer.Postcode?.toString().includes(filters.postcode))
              );
            });
            setFilteredConsumers(filtered);
          }}
        />

        {/* Main Content */}
        <main className={`p-6 overflow-auto ${darkMode ? 'text-dark-text' : ''}`}>
          <Box sx={{ p: 4 }}>
            <Typography variant="h4" gutterBottom fontWeight={700} color="primary">
              Consumers List
            </Typography>
            <Paper elevation={3} sx={{ borderRadius: 3, overflow: "hidden" }}>
              {loading ? (
                <Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: 200 }}>
                  <CircularProgress />
                </Box>
              ) : error ? (
                <Typography color="error" sx={{ p: 2 }}>{error}</Typography>
              ) : (
                <TableContainer>
                  <Table>
                    <TableHead sx={{ background: "linear-gradient(90deg, #1976d2 0%, #42a5f5 100%)" }}>
                      <TableRow>
                        <TableCell sx={{ color: "white", fontWeight: 600 }}>Consumer ID</TableCell>
                        <TableCell sx={{ color: "white", fontWeight: 600 }}>Postcode</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {filteredConsumers.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={2} align="center">No consumers found.</TableCell>
                        </TableRow>
                      ) : (
                        filteredConsumers.map((consumer) => {
                          const consumptionValue = consumer.maxConsumption;
                          let energyStatus;
                          if (consumptionValue < 0.5) {
                            energyStatus = "Low Usage";
                          } else if (consumptionValue < 1.2) {
                            energyStatus = "Medium Usage";
                          } else {
                            energyStatus = "High Usage";
                          }
                          return (
                            <TableRow key={consumer.Customer}>
                              <TableCell>{consumer.Customer}</TableCell>
                              <TableCell>{consumer.Postcode}</TableCell>
                            </TableRow>
                          );
                        })
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Paper>
          </Box>
        </main>
      </div>
    </div>
  );
};

export default ConsumerList;