import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Slider from "@mui/material/Slider";
import { BarChart } from "@mui/x-charts/BarChart";
import { useState } from "react";
import { axisClasses } from "@mui/x-charts";

function SimpleBarChart({ data }) {
  // Transform data: Count products per brand
  const brandCounts = data.reduce((acc, product) => {
    acc[product.brand] = (acc[product.brand] || 0) + 1;
    return acc;
  }, {});

  // Creating proper dataset array
  const dataset = Object.keys(brandCounts).map((brand) => ({
    brand: brand,
    count: brandCounts[brand],
  }))

  const topBrands = dataset.sort((a, b) => b.count - a.count).slice(0, 10);

  const brands = Object.keys(brandCounts).slice(0, 10);
  const counts = Object.values(brandCounts).slice(0, 10);

  const creatingListDictionary = () => {
    return brands.map((brand, index) => ({
      label: brand,
      data: [counts[index]],
    }));
  };

  console.log("Bar dict: ", creatingListDictionary());

  const [seriesNb, setSeriesNb] = useState(5);

  const handleSeriesNbChange = (e, newValue) => {
    if (typeof newValue !== "number") return;
    setSeriesNb(newValue);
  };

  const highlightScope = {
    highlight: "series",
    fade: "global",
  };

  const series = creatingListDictionary().map((s) => ({
    ...s,
    highlightScope,
  }));

  const chartSetting = {
    xAxis: [{
      scaleType: "band",
      dataKey: "brand",
      valueFormatter: (value) => value || 'Unknown Brand'
    }],
    yAxis: [
      {
        label: "brand",
      },
    ],
    series: [{
      dataKey: "count",
      label: "Product per Brand",
      highlightScope
    }],
    height: 300,
    sx: {
      [`& .${axisClasses.directionY} .${axisClasses.label}`]: {
        transform: "translateX(-10px)",
      },
    },
  };

  return (
    <div style={{ width: "100%" }}>
      <BarChart
        dataset={topBrands}
        barLabel="value"
        {...chartSetting}
      />
      <Typography
        className="text-primary"
        id="input-series-number"
        gutterBottom
      >
        Number of Series
      </Typography>
      <Slider
        values={seriesNb}
        onChange={handleSeriesNbChange}
        valueLabelDisplay="auto"
        min={1}
        max={10}
        aria-labelledby="input-series-number"
      />
    </div>
  );

}

export default SimpleBarChart;
