import React from "react";
import { Card, CardBody, CardHeader, Typography } from "@material-tailwind/react";
import { LineChart } from '@mui/x-charts/LineChart';

export default function BasicLineChart({ products_data }) {
  console.log(products_data);

  const productNameList = products_data.slice(0, 10).map((product) => product.name)
  const productIdList = products_data.slice(0, 10).map((product) => product.product_id)
  const productAvailabilityList = products_data.slice(0, 10).map((product) => parseFloat(product.disponivel))

  console.log(productIdList);
  console.log(productNameList);
  console.log(productAvailabilityList);

  return (
    <LineChart
      xAxis={[{ data: productAvailabilityList }]}
      series={[
        {
          data: productIdList, label: 'Product ID list',
        },
      ]}
      width={500}
      height={300}
    />
  );
}