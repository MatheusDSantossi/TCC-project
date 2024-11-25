import logo from '../assets/logo.svg';

function ProductRegistration() {
  return (
    <div className="flex justify-center items-center min-h-screen p-4">
      <div className="w-full max-w-lg bg-primary p-8 shadow-xl rounded-lg">
        <div className="flex mb-5 justify-center items-center">
          <img className='w-20 mr-4' src={logo} alt="" />
          <h2 className="text-2xl left-40 mb-6 font-semibold text-center">Products Registration</h2>
        </div>
        <form action="" className="grid grid-cols-1 gap-6">
          {/* Product Name */}
          <div>
            <label htmlFor="productName" className="block text-sm font-semibold text-white">Product Name</label>
            <input
              type="text"
              id="productName"
              name="productName" required className="mt-1 p-2 w-full border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
            />
          </div>

          {/* Product Section */}
          <div>
            <label htmlFor="productSection" className="block text-sm font-semibold text-white">Product Section</label>
            <input
              type="text"
              id="productSection" name="productSection" required className="mt-1 p-2 w-full border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
            />
          </div>
          {/* Product Price */}
          <div>
            <label htmlFor="productPrice" className="block text-sm font-semibold text-white">Product Price</label>
            <input
              type="text"
              id="productPrice" name="productPrice" required className="mt-1 p-2 w-full border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
            />
          </div>
          {/* Prouct Metric Unity */}
          <div>
            <label htmlFor="productMetricUnity" className="block text-sm font-semibold text-white">Product Metric Unity</label>
            <input type="text" id="productMetricUnity"
              name="ProductRegistration" required className="mt-1 p-2 w-full border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
            />
          </div>
          {/* Product Availability */}
          <div>
            <label htmlFor="productAvailability" className="block text-sm font-semibold text-white">Product Availability</label>
            <div className="flex">
              <label htmlFor="productAvailability" className="block text-sm font-medium text-white mr-3">Yes</label>
              <input
                type="radio"
                id="productAvailability"
                name="productAvailability" value="Yes" required />
            </div>
            <div className="flex">
              <label htmlFor="productAvailability" className="block text-sm font-medium text-white mr-3">No&nbsp;&nbsp;</label>
              <input
                type="radio"
                id="productAvailability" 
                name="productAvailability" 
                value="No" required />
            </div>
          </div>

          <button className="mt-4 bg-yellow-400 hover:bg-yellow-300 text-black font-bold py-2 px-4 rounded-lg transition duration-200" type="submit">Register</button>
        </form>
      </div>
    </div>
  )
}

export default ProductRegistration