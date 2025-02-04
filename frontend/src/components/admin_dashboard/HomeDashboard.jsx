import React, { useEffect, useState } from 'react'
import 
{ BsFillArchiveFill, BsFillGrid3X3GapFill, BsPeopleFill, BsFillBellFill}
 from 'react-icons/bs'
import BasicPieChart from './charts/PieChart'
import BarChart from './charts/BarChart'
import AxiosInstance from '../AxiosInstance';
import ProductsTable from './charts/Table';

function HomeDashboard() {
  
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);

  const totalProducts = products.length;
  const availableProducts = products.filter((p) => parseFloat(p.disponivel) > 0).length;
  const averagePrice = products.reduce((sum, product) => sum + parseFloat(product.preco_venda || 0), 0) / totalProducts || 0;

  const [numberOfRows, setNumberofRows] = useState(10);

  const fetchProductsData = () => {

    AxiosInstance.get(`api/products/`)
      // .then((response) => setMenuOptions(response.data))
      .then((response) => {
        setProducts(response.data)
        setLoading(false);
      })
      .catch((error) => console.error("Error fetching product data: ", error));
  };

  useEffect(() => {
    fetchProductsData();
  }, []);

  return (
    <div className="flex flex-col p-6 space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <div className="p-4 text-center bg-white rounded-lg shadow-card">
          <h3 className="text-lg text-primary">Total Products</h3>
          <p className="text-2xl font-bold text-primary">{loading ? 'Loading...' : totalProducts}</p>
        </div>
        <div className="p-4 text-center bg-white rounded-lg shadow-card">
          <h3 className="text-lg text-primary">Available Products</h3>
          <p className="text-2xl font-bold text-primary">{loading ? 'Loading...' : availableProducts}</p>
        </div>
        <div className="p-4 text-center bg-white rounded-lg shadow-card">
          <h3 className="text-lg text-primary">Average Price</h3>
          <p className="text-2xl font-bold text-primary">{loading ? 'Loading...' : `${averagePrice.toFixed(2)}`}</p>
        </div>
        <div className="p-4 text-center bg-white rounded-lg shadow-card">
          <h3 className="text-lg text-primary">Low Stock (-10)</h3>
          <p className="text-2xl font-bold text-primary">{loading ? 'Loading...' : products.filter((p) => parseFloat(p.disponivel) < 10).length}</p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="w-full p-6 text-center bg-white rounded-lg shadow-card">
          <p className="text-2xl font-medium text-primary">Products by brand</p>
          <BarChart data={products} />
        </div>
        <div className="w-full p-6 text-center bg-white rounded-lg shadow-card">
          <p className="text-2xl font-medium text-primary">Stock Availability by Section</p>
          <BasicPieChart products_data={products} />
        </div>
      </div>

      {/* Recent Products */}
      <div className="w-full p-6 bg-white rounded-lg shadow-card text-primary">
        <h3 className="mb-4 text-2xl font-medium text-primary">Products</h3>
          
            {loading ? (
              <div className="text-center">
              Loading...
              </div>
            ) : (
              <ProductsTable products_data={products} />
          )}
      </div>
    </div>
  );
}

export default HomeDashboard
