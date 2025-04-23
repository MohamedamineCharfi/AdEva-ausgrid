import { predictConsumption } from "./api";

export const handlePredictionLogic = async ({
  records,
  consumerId,
  setNotification,
  setPredictionData,
}) => {
  // Fixed range: June 1, 2013 to June 30, 2013
  const startDate = new Date("2013-06-01");
  const endDate = new Date("2013-06-30");

  const june2013Data = records.filter((record) => {
    const recordDate = new Date(record.date);
    return (
      record.ConsumerId === consumerId &&
      recordDate >= startDate &&
      recordDate <= endDate
    );
  });

  if (june2013Data.length === 0) {
    setNotification("No data available for June 2013 to predict.");
    return;
  }

  try {
    const prediction = await predictConsumption(june2013Data);
    setPredictionData(prediction);
    setNotification(null);
  } catch (error) {
    setNotification(error.message);
  }
};
