// src/services/predictHelper.js
import { predictConsumption } from "./api";

export const handlePredictionLogic = async ({
  /*records,*/
  consumerId,
  setNotification,
  setPredictionData,
}) => {
  if (!consumerId || consumerId === "All") {
    return setNotification("You must select a consumer before predicting.");
  }

  const month = "2013-06";

  try {
    const forecast = await predictConsumption({ consumerId, month });
    // build simple Day-labels for your 7-day chart
    const labels = forecast.map((_, i) => `Day ${i + 1}`);
    setPredictionData({ labels, values: forecast });
    setNotification(null);
  } catch (err) {
    setNotification(err.message);
  }
};
