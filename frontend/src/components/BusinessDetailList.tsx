import { useEffect, useState } from "react";
import AxiosInstance from "./AxiosInstance";
import BusinessDetailsTable from "./admin_dashboard/charts/BusinessDetailsTable";
import { useException } from "../context/ExceptionContext";
import { ExceptionVariant } from "../exceptions/ExceptionVariantsTypes";

const BusinessDetailList = () => {
  const [businessDetail, setBusinessDetail] = useState([]);
  const [refreshKey, setRefreshKey] = useState(0);

  const { showException } = useException();

  const fetchBusinessDetailData = () => {
    AxiosInstance.get(`api/business-detaila/`)
      // .then((response) => setMenuOptions(response.data))
      .then((response) => {
        console.log("Response: ", response);
        console.log("Response data: ", response.data);
        setBusinessDetail(response.data);
      })
      .catch((error) => {
        console.error("Error fetching business detail data: ", error);

        showException({
          text: "Falha ao carregar detalhes da empresa",
          error: error,
          variant: ExceptionVariant.ERROR,
        });
      });
  };

  // Adding a refresh function
  const refreshBusinessDetail = () => {
    setRefreshKey((prev) => prev + 1);
  };

  useEffect(() => {
    fetchBusinessDetailData();
  }, [refreshKey]);

  console.log("First businessDetail: ", businessDetail);

  return (
    <div>
      <BusinessDetailsTable
        business_detail_data={businessDetail}
        onBusinessDetailUpdate={refreshBusinessDetail}
      />
    </div>
  );
};

export default BusinessDetailList;
