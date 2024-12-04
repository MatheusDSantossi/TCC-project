import React from 'react'
import 
{ BsFillArchiveFill, BsFillGrid3X3GapFill, BsPeopleFill, BsFillBellFill}
 from 'react-icons/bs'
import LineChart from './charts/LineChart'
import BarChart from './charts/BarChart'

function HomeDashboard() {
  return (
    <div className="flex flex-col p-6 space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
        <div className="bg-white shadow-card p-4 rounded-lg text-center">
          <h3 className="text-lg font-medium">Total Products</h3>
          <p className="text-2xl font-bold text-primary">4805</p>
        </div>
        <div className="bg-white shadow-card p-4 rounded-lg text-center">
          <h3 className="text-lg font-medium">Total Products</h3>
          <p className="text-2xl font-bold text-primary">4805</p>
        </div>
        <div className="bg-white shadow-card p-4 rounded-lg text-center">
          <h3 className="text-lg font-medium">Total Products</h3>
          <p className="text-2xl font-bold text-primary">4805</p>
        </div>
        <div className="bg-white shadow-card p-4 rounded-lg text-center">
          <h3 className="text-lg font-medium">Total Products</h3>
          <p className="text-2xl font-bold text-primary">4805</p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="bg-white shadow-card p-6 rounded-lg text-center">
          <p className="text-lg font-medium">Bar Chart Placeholder</p>
          <BarChart />
        </div>
        <div className="bg-white shadow-card p-6 rounded-lg text-center">
          <p className="text-lg font-medium">Line Chart Placeholder</p>
          <LineChart />
        </div>
      </div>

      {/* Recent Orders */}
      <div className="bg-white shadow-card p-6 rounded-lg">
        <h3 className="text-lg font-medium mb-4">Recent Orders</h3>
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b">
              <th className="p-2">Product</th>
              <th className="p-2">Product</th>
              <th className="p-2">Product</th>
              <th className="p-2">Product</th>
              <th className="p-2">Product</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b">
              <td className="p-2">Café santa clara</td>
              <td className="p-2">Café santa clara</td>
              <td className="p-2">Café santa clara</td>
              <td className="p-2">Café santa clara</td>
              <td className="p-2">Café santa clara</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default HomeDashboard
