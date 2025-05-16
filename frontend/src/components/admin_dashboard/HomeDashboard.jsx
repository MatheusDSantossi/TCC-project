import React, { useEffect, useState } from "react";
import {
  BsFillArchiveFill,
  BsFillGrid3X3GapFill,
  BsPeopleFill,
  BsFillBellFill,
} from "react-icons/bs";
import BasicPieChart from "./charts/PieChart";
import SimpleBarChart from "./charts/BarChart";
import AxiosInstance from "../AxiosInstance";
import ProductsTable from "./charts/ProductsTable";
import { useNavigate } from "react-router-dom";

function HomeDashboard() {
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const totalProducts = products.length;
  const availableProducts = products.filter(
    (p) => parseFloat(p.disponivel) > 0
  ).length;
  const averagePrice =
    products.reduce(
      (sum, product) => sum + parseFloat(product.preco_venda || 0),
      0
    ) / totalProducts || 0;

  const fetchProductsData = () => {
    AxiosInstance.get(`api/products/`)
      // .then((response) => setMenuOptions(response.data))
      .then((response) => {
        setProducts(response.data);
        setLoading(false);
      })
      .catch((error) => {
        AxiosInstance.post(`logoutall/`, {}).then(() => {
          // Clear user session data and navigate to login pagel
          localStorage.removeItem("Token");
    
          // Navigate to login page
          navigate("/login");
        });
        console.error("Error fetching product data: ", error);
      });
  };

  useEffect(() => {
    fetchProductsData();
  }, []);

  return (
    <div className="flex flex-col p-6 space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <div className="p-4 text-center bg-white rounded-lg shadow-card">
          <h3 className="text-lg text-primary">Quantidade Total de Produtos</h3>
          <p className="text-2xl font-bold text-primary">
            {loading ? "Loading..." : totalProducts}
          </p>
        </div>
        <div className="p-4 text-center bg-white rounded-lg shadow-card">
          <h3 className="text-lg text-primary">
            Quantidade de Produtos Disponíveis
          </h3>
          <p className="text-2xl font-bold text-primary">
            {loading ? "Loading..." : availableProducts}
          </p>
        </div>
        <div className="p-4 text-center bg-white rounded-lg shadow-card">
          <h3 className="text-lg text-primary">Preço Médio</h3>
          <p className="text-2xl font-bold text-primary">
            {loading ? "Loading..." : `${averagePrice.toFixed(2)}`}
          </p>
        </div>
        <div className="p-4 text-center bg-white rounded-lg shadow-card">
          <h3 className="text-lg text-primary">Produtos com Estoque Baixo</h3>
          <p className="text-2xl font-bold text-primary">
            {loading
              ? "Loading..."
              : products.filter((p) => parseFloat(p.disponivel) < 10).length}
          </p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="w-full p-6 text-center bg-white rounded-lg shadow-card">
          <p className="text-2xl font-medium text-primary">
            Produtos por Marca
          </p>
          <SimpleBarChart data={products} />
        </div>
        <div className="w-full p-6 text-center bg-white rounded-lg shadow-card">
          <p className="text-2xl font-medium text-primary">
            Produtos Disponíveis por Seção
          </p>
          <BasicPieChart products_data={products} />
        </div>
      </div>

      {/* Recent Products */}
      {/* <div className="flex flex-col items-center justify-center w-full p-6 bg-white rounded-lg shadow-card text-primary">
        <h3 className="mb-4 text-2xl font-medium text-primary">Produtos</h3>
          
            {loading ? (
              <div className="text-center">
              Loading...
              </div>
            ) : (
              <ProductsTable products_data={products} />
          )}
      </div> */}
    </div>
  );
}

export default HomeDashboard;
