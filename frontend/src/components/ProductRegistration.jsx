import { useEffect, useState } from "react";
import logo from "../assets/logo.svg";
import { getCSRFToken } from "../utils/CSRFToken";
import AxiosInstance from "./AxiosInstance";

// function getCSRFToken() {
//   const name = 'csrftoken';
//   const cookieValue = document.cookie
//       .split('; ')
//       .find((row) => row.startsWith(name))
//       ?.split('=')[1];
//   return cookieValue || '';
// }

function ProductRegistration() {
  const [productName, setProductName] = useState("");
  const [productBrand, setProductBrand] = useState("");
  const [productQuantityFormat, setProductQuantityFormat] = useState("");
  const [productSection, setProductSection] = useState(0);
  const [productPrice, setProductPrice] = useState("");
  const [productMetricUnity, setProductMetricUnity] = useState("");
  const [productAvailability, setProductAvailibility] = useState("");

  const [suggestions, setSuggestions] = useState([]);

  const GetProductList = (productName) => {
    if (productName.length > 3) {
      AxiosInstance.get(
        `api/products/?search=${productName}`
      )
        .then((response) => {
          
          console.log(response.data);
          setSuggestions(response.data);
          
        })
        .catch((err) => {
          console.error("Error fetching product suggestions: ", err);
          setSuggestions([]);
        });
    } else {
      setSuggestions([]);
    }
  };

  useEffect(() => {
    GetProductList(productName);
  }, [productName]);

  const handleProductSelection = (selectedProductName) => {
    const selectedProduct = suggestions.find((product) => product.name === selectedProductName);

    if (selectedProduct) {
      setProductBrand(selectedProduct.brand);
      setProductQuantityFormat(selectedProduct.quantity);
    }
  };

  const registerProduct = () => {
    
    AxiosInstance.post(`register-product/post/`, {
        productName: productName.trim(),
        productBrand: productBrand.trim(),
        productQuantityFormat: productQuantityFormat.trim(),
        productSection: Number(productSection),
        productPrice: productPrice,
        productMetricUnity: productMetricUnity.trim(),
        productAvailability: productAvailability,
      })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to register the product");
        }
        return response.json();
      })
      .then((data) => {
        console.log("Product registered succesfully: ", data);
        alert("Product registered succesfully!");
      })
      .catch((error) => {
        console.error("Error registering the product: ", error);
      });
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4">
      <div className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary">
        <div className="flex items-center justify-center mb-5">
          <img className="w-20 mr-4" src={logo} alt="" />
          <h2 className="mb-6 text-2xl font-semibold text-center left-40">
            Products Registration
          </h2>
        </div>
        <form action="" className="grid grid-cols-1 gap-6">
          {/* # divide the product name into three parts (name, brand, quantity) */}
          {/* Product Name */}
          <div>
            <label
              htmlFor="productName"
              className="block text-sm font-semibold text-white"
            >
              Product Name
            </label>
            <input
              type="text"
              id="productName"
              name="productName"
              required
              className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productName}
              list="productsSuggestions"
              onChange={(e) => setProductName(e.target.value)}
              onBlur={(e) => handleProductSelection(e.target.value)} // Triggered when a product is selected
            />
            <datalist id="productsSuggestions">
              {suggestions.map((product, index) => (
                <option key={index} value={product.name}>
                  {product.name}
                </option>
              ))}
            </datalist>
          </div>

          {/* Product brand */}
          <div>
            <label
              htmlFor="productBrand"
              className="block text-sm font-semibold text-white"
            >
              Product Brand
            </label>
            <input
              type="text"
              id="productBrand"
              name="productBrand"
              required
              className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productBrand}
              onChange={(e) => setProductBrand(e.target.value)}
              placeholder="Néscau"
            />
          </div>

          {/* Product Quantity Format */}
          <div>
            <label
              htmlFor="productQuantityFormat"
              className="block text-sm font-semibold text-white"
            >
              Product Quantity Format
            </label>
            <input
              type="text"
              id="productQuantityFormat"
              name="productQuantityFormat"
              required
              className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productQuantityFormat}
              onChange={(e) => setProductQuantityFormat(e.target.value)}
              placeholder="6x12"
            />
          </div>

          {/* Product Section */}
          <div>
            <label
              htmlFor="productSection"
              className="block text-sm font-semibold text-white"
            >
              Product Section
            </label>
            <input
              type="number"
              id="productSection"
              name="productSection"
              required
              className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productSection}
              onChange={(e) => setProductSection(e.target.value)}
              placeholder="Seção 1"
            />
          </div>
          {/* Product Price */}
          <div>
            <label
              htmlFor="productPrice"
              className="block text-sm font-semibold text-white"
            >
              Product Price
            </label>
            <input
              type="text"
              id="productPrice"
              name="productPrice"
              required
              className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productPrice}
              onChange={(e) => setProductPrice(e.target.value)}
            />
          </div>
          {/* Prouct Metric Unity */}
          <div>
            <label
              htmlFor="productMetricUnity"
              className="block text-sm font-semibold text-white"
            >
              Product Metric Unity
            </label>
            <input
              type="text"
              id="productMetricUnity"
              name="productMetricUnity"
              required
              className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productMetricUnity}
              onChange={(e) => setProductMetricUnity(e.target.value)}
              placeholder="UN"
            />
          </div>
          {/* Product Availability */}
          <div>
            <label
              htmlFor="productAvailability"
              className="block text-sm font-semibold text-white"
            >
              Product Availability
            </label>
            <input
              type="text"
              id="productAvailability"
              name="productAvailability"
              required
              className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productAvailability}
              onChange={(e) => setProductAvailibility(e.target.value)}
              placeholder="100"
            />
          </div>

          <button
            className="px-4 py-2 mt-4 font-bold text-black transition duration-200 bg-yellow-400 rounded-lg hover:bg-yellow-300"
            type="submit"
            onClick={registerProduct}
          >
            Register
          </button>
        </form>
      </div>
    </div>
  );
}

export default ProductRegistration;
