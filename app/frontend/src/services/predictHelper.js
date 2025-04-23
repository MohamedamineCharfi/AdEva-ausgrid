// src/services/predictHelper.js
import { predictConsumption } from "./api";

export const handlePredictionLogic = async ({
  records,
  consumerId,
  setNotification,
  setPredictionData,
}) => {
  if (!consumerId) {
    return setNotification("You must select a consumer before predicting.");
  }

  // get June of the last year in your dataset:
  const targetMonth = "2013-06";  
  const [year, month] = targetMonth.split("-").map(Number);

  const juneData = records.filter((r) => {
    const d = new Date(r.date);
    return (
      Number(r.Customer) === Number(consumerId) &&
      d.getFullYear() === year &&
      d.getMonth() + 1 === month
    );
  });

  if (!juneData.length) {
    return setNotification(`No data available for ${targetMonth} to predict.`);
  }

  try {
    const prediction = await predictConsumption({ consumerId, month: targetMonth });
    setPredictionData(prediction);
    setNotification(null);
  } catch (err) {
    setNotification(err.message);
  }
};
