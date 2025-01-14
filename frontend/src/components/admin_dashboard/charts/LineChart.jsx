import {
    Card,
    CardBody,
    CardHeader,
    Typography
} from '@material-tailwind/react';
import Chart from 'react-apexcharts';
// import { Square3Stack3DIcon } from '@heroicons/react/24/outline';

function LineChart({ data }) {
  // Transform data: Get the stock availability of the first 10 products
  const productNames = data.slice(0, 10).map((product) => product.name);
  const stockLevels = data.slice(0, 10).map((product) => parseFloat(product.disponivel));

  console.log("Data: ", data);
  console.log("Product Names: ", productNames);
  console.log("Stock Levels: ", stockLevels);

  // Chart configuration
  const chartConfig = {
    type: "line",
    height: 240,
    series: [
      {
        name: "Stock Availability",
        data: stockLevels,
      },
    ],
    options: {
      chart: {
        toolbar: {
          show: false,
        },
      },
      title: {
        show: "",
      },
      dataLabels: {
        enabled: false,
      },
      colors: ["#020617"],
      stroke: {
        // lineCap: "round",
        curve: "smooth",
      },
      // markers: {
      //   size: 0,
      // },
      xaxis: {
        categories: productNames,
        labels: {
          style: {
            colors: "#616161",
            fontSize: "12px",
            fontFamily: "inherit",
            fontWeight: 400,
          },
        },
        // axisTicks: {
        //   show: false,
        // },
        // axisBorder: {
        //   show: false,
        // },
      },
      yaxis: {
        labels: {
          style: {
            colors: "#616161",
            fontSize: "12px",
            fontFamily: "inherit",
            fontWeight: 400,
          },
        },
      },
      grid: {
        show: true,
        borderColor: "#dddddd",
        strokeDashArray: 5,
        xaxis: {
          lines: {
            show: true,
          },
        },
        padding: {
          top: 5,
          right: 20,
        },
      },
      fill: {
        opacity: 0.8,
      },
      tooltip: {
        theme: "dark",
      },
    },
  };

  return (
    <Card>
        <CardHeader
            floated={false}
            shadow={false}
            color="transparent"
            className="flex flex-col gap-4 rounded-none md:flex-row md:items-center"
        >
            <div className="p-5 text-white bg-gray-900 rounded-lg w-a">
                <p>TESTE</p>
            </div>
            <div>
                <Typography variant="h" color="blue-gray">
                    Stock Availability
                </Typography>
                <Typography 
                variant="small" color="gray"
                className="max-w-sm font-normal">
                    Top 10 Products
                </Typography>
            </div>
        </CardHeader>
        <CardBody className="px-2 pb-0">
            <Chart {...chartConfig} />
        </CardBody>
    </Card>
  )
}

export default LineChart