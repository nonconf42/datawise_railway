import React, { useState, useEffect } from 'react';

const StatisticsDisplay = ({ statistics }) => {
  const [stats, setStats] = useState([]);
  
  useEffect(() => {
    // Parse statistics if they're provided as a string
    if (typeof statistics === 'string') {
      const lines = statistics.split('\n').filter(line => line.trim());
      const parsedStats = lines.map(line => line.trim());
      setStats(parsedStats);
    } else if (Array.isArray(statistics)) {
      setStats(statistics);
    }
  }, [statistics]);

  if (!stats.length) {
    return <div className="text-gray-500 italic">No statistics available</div>;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {stats.map((stat, index) => (
        <div 
          key={index}
          className="bg-green-50 bg-opacity-80 p-4 rounded-lg border border-green-200 shadow-sm"
        >
          <p className="text-green-800 text-sm">{stat}</p>
        </div>
      ))}
    </div>
  );
};

export default StatisticsDisplay;