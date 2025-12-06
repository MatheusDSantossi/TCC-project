import { useEffect, useState } from "react";
import ProductsTable from "./admin_dashboard/charts/ProductsTable";
import AxiosInstance from "./AxiosInstance";

const ProductList = () => {
  const [products, setProducts] = useState([]);
  const [refreshKey, setRefreshKey] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  const fetchProductsData = () => {
    setIsLoading(true);
    AxiosInstance.get(`api/products/`)
      // .then((response) => setMenuOptions(response.data))
      .then((response) => {
        setProducts(response.data);
      })
      .catch((error) => console.error("Error fetching product data: ", error))
      .finally(() => setIsLoading(false));
  };

  // Adding a refresh function
  const refreshProducts = () => {
    setRefreshKey((prev) => prev + 1);
  };

  useEffect(() => {
    fetchProductsData();
  }, [refreshKey]);

  console.log("First Products: ", products);

  return (
    <div>
      <ProductsTable
        products_data={products}
        onProductUpdate={refreshProducts}
        isLoading={isLoading}
      />
    </div>
  );
};

export default ProductList;
