import React, { useEffect, useState } from 'react'
import 
{ BsFillArchiveFill, BsFillGrid3X3GapFill, BsPeopleFill, BsFillBellFill}
 from 'react-icons/bs'
import LineChart from './charts/LineChart'
import BarChart from './charts/BarChart'
import AxiosInstance from '../AxiosInstance';

function HomeDashboard() {
  
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);

  const totalProducts = products.length;
  const availableProducts = products.filter((p) => parseFloat(p.disponivel) > 0).length;
  const averagePrice = products.reduce((sum, product) => sum + parseFloat(product.preco_venda || 0), 0) / totalProducts || 0;

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
        <div className="p-6 text-center bg-white rounded-lg shadow-card">
          <p className="text-2xl font-medium text-primary">Products by brand</p>
          <BarChart data={products} />
        </div>
        <div className="p-6 text-center bg-white rounded-lg shadow-card">
          <p className="text-2xl font-medium text-primary">Stock Availability</p>
          <LineChart data={products} />
        </div>
      </div>

      {/* Recent Products */}
      <div className="p-6 bg-white rounded-lg shadow-card text-primary">
        <h3 className="mb-4 text-lg font-medium">Recent Products</h3>
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b">
              <th className="p-2">Product</th>
              <th className="p-2">Brand</th>
              <th className="p-2">Price</th>
              <th className="p-2">Stock</th>
              <th className="p-2">Unit</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan="5" className="text-center">Loading...</td>
              </tr>
            ) : (products.slice(0, 10).map((product) => (
            <tr className="border-b">
              <td className="p-2">{product.name}</td>
              <td className="p-2">{product.brand}</td>
              <td className="p-2">{product.preco_venda}</td>
              <td className="p-2">{product.disponivel}</td>
              <td className="p-2">{product.unidade_medida}</td>
            </tr>
            ))
          )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default HomeDashboard
