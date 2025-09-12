import React, { useEffect, useState } from "react";
import { PieChart } from "@mui/x-charts/PieChart";

function BasicPieChart({ products_data }) {
  if (!products_data || products_data.length === 0) {
    return <div>No data to display</div>;
  }

  const productSectionList = products_data
    .slice(0, 10)
    .map((product) => product.section);
  const productIdList = products_data
    .slice(0, 10)
    .map((product) => product.product_id);
  const productAvailabilityList = products_data
    .slice(0, 10)
    .map((product) => parseFloat(product.stock));

  const [productSectionListV2, setProductSectionListV2] = useState([
    ...new Set(productSectionList),
  ]);

  console.log(productIdList);
  console.log("Section : ", productSectionList);
  console.log("Section v2 : ", productSectionListV2);
  console.log(productAvailabilityList);

  const creatingListDict = () => {
    const listDict = [];

    productSectionListV2.forEach((section, index) => {
      console.log("Index: ", index);
      console.log("Section: ", section);
      const dictionary = {
        id: index,
        value: productAvailabilityList[index],
        label: section,
      };
      listDict.push(dictionary);
    });

    let sortedAscending = Object.values(listDict).sort((a, b) => {
      return b.value - a.value;
    });

    // console.log("Sorted ascending: ", sortedAscending);
    return sortedAscending;
  };
  console.log("Dictionary: ", creatingListDict());

  const series = {
    data: creatingListDict(),
    highlightScope: { fade: "global", highlight: "item" },
    faded: { innerRadius: 30, additionalRadius: -30, color: "gray" },
  };

  return (
    <div className="w-full h-64">
      <PieChart 
      series={[series]} 
      width={400} 
      height={200} 
      sx={{
        width: '100%',
        height: '100%',
        maxWidth: '400px',
        maxHeight: '200px'
      }}
      />
    </div>
  );
}

export default BasicPieChart;
