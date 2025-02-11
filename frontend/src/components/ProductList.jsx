import { useEffect, useState } from "react";
import ProductsTable from "./admin_dashboard/charts/Table";
import AxiosInstance from "./AxiosInstance";

const ProductList = () => {
  const [products, setProducts] = useState([]);
  const [refreshKey, setRefreshKey] = useState(0);

  const fetchProductsData = () => {
    AxiosInstance.get(`api/products/`)
      // .then((response) => setMenuOptions(response.data))
      .then((response) => {
        setProducts(response.data);
      })
      .catch((error) => console.error("Error fetching product data: ", error));
  };

  // Adding a refresh function
  const refreshProducts = () => {
    setRefreshKey(prev => prev + 1);
  };

  useEffect(() => {
    fetchProductsData();
  }, [refreshKey]);

  console.log("First Products: ", products)

  return (
    <div>
      <ProductsTable 
      products_data={products} 
      onProductUpdate={refreshProducts}
      />
    </div>
  );
};

export default ProductList;
