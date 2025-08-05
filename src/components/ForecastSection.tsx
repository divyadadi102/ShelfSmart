import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Users, DollarSign, Package, Star, CloudRain } from "lucide-react";
import { useState } from "react";
import WeatherCard from "./WeatherCard";
import TexasMap from "./TexasMap";
import { useEffect, } from "react";

type PredictionPeriod = 'today' | 'tomorrow' | 'nextWeek';
type ForecastData = {
    allProducts: { name: string; expected: number; color?: string }[];
    expectedTraffic: number;
    expectedRevenue: number;
    inventoryTurnover: number;
    profitMargin: number;
    weatherImpact: string;
  };

const ForecastSection = () => {
  const [predictionPeriod, setPredictionPeriod] = useState<PredictionPeriod>('today');
  const [predictionData, setPredictionData] = useState<Record<PredictionPeriod, ForecastData> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // localhost URL: http://localhost:8000/api/forecast
        const res = await fetch("/api/api/forecast", {
          headers: {
            Authorization: `Bearer ${localStorage.getItem("access_token")}`,
          },
        });

        if (!res.ok) throw new Error("Failed to fetch forecast data");
        const data = await res.json();
        setPredictionData(data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

  fetchData();
}, []);

  const getPredictionTitle = () => {
    switch (predictionPeriod) {
      case 'today': return "Today's Forecasts";
      case 'tomorrow': return "Tomorrow's Forecasts";
      case 'nextWeek': return "Next Week's Forecasts";
      default: return "Forecasts";
    }
  };
  if (loading) {
  return <p className="text-center text-gray-500">Loading forecasts...</p>;
  }

  if (!predictionData) {
    return <p className="text-center text-red-500">No forecast data available.</p>;
  }
  const currentPrediction = predictionData[predictionPeriod];

  return (
    <div className="mb-12">
      {/* Prediction Period Selector */}
      <div className="mb-6">
        <div className="flex space-x-2">
          <Button
            variant={predictionPeriod === 'today' ? 'default' : 'outline'}
            onClick={() => setPredictionPeriod('today')}
            className="flex items-center space-x-2"
          >
            <span>Today's Forecast</span>
          </Button>
          <Button
            variant={predictionPeriod === 'tomorrow' ? 'default' : 'outline'}
            onClick={() => setPredictionPeriod('tomorrow')}
            className="flex items-center space-x-2"
          >
            <span>Tomorrow's Forecast</span>
          </Button>
          <Button
            variant={predictionPeriod === 'nextWeek' ? 'default' : 'outline'}
            onClick={() => setPredictionPeriod('nextWeek')}
            className="flex items-center space-x-2"
          >
            <span>Next Week's Forecast</span>
          </Button>
        </div>
      </div>

      <div className="flex items-center space-x-2 mb-6">
        <div className="w-2 h-8 bg-gradient-to-b from-blue-500 to-purple-500 rounded-full"></div>
        <h2 className="text-2xl font-bold text-gray-900">{getPredictionTitle()}</h2>
      </div>

      {/* Holiday/Weekend Effect Banner */}
      {predictionPeriod === 'today' && (
        <Card className="mb-6 border-orange-200 bg-orange-50">
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <Star className="w-6 h-6 text-orange-600" />
              <div>
                <h3 className="font-semibold text-orange-800">Weekend Effect Active</h3>
                <p className="text-sm text-orange-700">Expected 25% increase in foot traffic due to weekend shopping patterns</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {predictionPeriod === 'nextWeek' && (
        <Card className="mb-6 border-green-200 bg-green-50">
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <Star className="w-6 h-6 text-green-600" />
              <div>
                <h3 className="font-semibold text-green-800">Weekly Planning Mode</h3>
                <p className="text-sm text-green-700">Plan your inventory orders in advance based on next week's demand forecast</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* All Products Forecast - Bar Chart */}
      <div className="mb-6">
        <Card className="shadow-lg">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Package className="w-5 h-5 text-green-600" />
              <span>Complete Product Sales Forecast</span>
            </CardTitle>
            <CardDescription>
              {predictionPeriod === 'nextWeek' ? 'Expected sales for all products next week' : `Expected sales for all products ${predictionPeriod}`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={currentPrediction.allProducts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="expected" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Forecast Cards Grid - Horizontal Layout */}
      <div className="grid grid-cols-4 gap-6 mb-6">
        <WeatherCard />

        <Card className="shadow-lg">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center space-x-2 text-sm">
              <Users className="w-4 h-4 text-purple-600" />
              <span>Expected Traffic</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <p className="text-2xl font-bold text-purple-600 mb-1">{currentPrediction.expectedTraffic}</p>
              <p className="text-xs text-gray-600">
                {predictionPeriod === 'nextWeek' ? 'customers expected' : 'customers expected'}
              </p>
              <Badge className="mt-1 bg-green-100 text-green-800 text-xs">
                {predictionPeriod === 'nextWeek' ? '+20% vs this week' : '+12% vs yesterday'}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="shadow-lg">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center space-x-2 text-sm">
              <DollarSign className="w-4 h-4 text-green-600" />
              <span>Revenue Forecast</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600 mb-1">${currentPrediction.expectedRevenue.toLocaleString()}</p>
              <p className="text-xs text-gray-600">
                {predictionPeriod === 'nextWeek' ? 'expected next week' : `expected ${predictionPeriod}`}
              </p>
              <Badge className="mt-1 bg-green-100 text-green-800 text-xs">
                {predictionPeriod === 'nextWeek' ? '+22% vs this week' : '+18% vs avg'}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <TexasMap />
      </div>
    </div>
  );
};

export default ForecastSection;
