import {
  Card,
  CardBody,
  CardHeader,
  Typography,
} from "@material-tailwind/react";
import Chart from "react-apexcharts";

function BarChart({ data }) {
  // Transform data: Count products per brand
  const brandCounts = data.reduce((acc, product) => {
    acc[product.brand] = (acc[product.brand] || 0) + 1;
    return acc;
  }, {});

  const brands = Object.keys(brandCounts).slice(0, 10);
  const counts = Object.values(brandCounts).slice(0, 10);

  const chartConfig = {
    type: "bar",
    height: 240,
    series: [
      {
        name: "Products",
        data: counts,
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
      plotOptions: {
        bar: {
          columnWidth: "40%",
          borderRadius: 2,
        },
      },
      xaxis: {
        axisTicks: {
          show: false,
        },
        axisBorder: {
          show: false,
        },
        labels: {
          style: {
            colors: "#616161",
            fontSize: "12px",
            fontFamily: "inherit",
            fontWeight: 400,
          },
        },
        categories: brands,
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
    <div className="w-full h-64">
      <Card className="w-full h-full">
        <CardHeader
          floated={false}
          shadow={false}
          color="transparent"
          className="flex gap-4 rounded-none flex-clo md:flex-row md:items-center"
        >
          <div className="p-5 text-white bg-gray-900 rounded-lg w-max">
            TESTE
          </div>
          <div>
            <Typography variant="h6" color="blue-gray">
              Products by Brand
            </Typography>
            {/* <Typography variant="small"
                    color="gray" className="max-w-sm font-normal">
                        12,345
                    </Typography> */}
          </div>
        </CardHeader>
        <CardBody className="w-full h-full px-2 pb-0">
          <div className="w-full h-48 mb-10">
            <Chart {...chartConfig} />
          </div>
        </CardBody>
      </Card>
    </div>
  );
}

export default BarChart;
