import { useEffect, useState } from 'react';
import logo from '../assets/logo.svg';
import { getCSRFToken } from '../utils/CSRFToken';

// function getCSRFToken() {
//   const name = 'csrftoken';
//   const cookieValue = document.cookie
//       .split('; ')
//       .find((row) => row.startsWith(name))
//       ?.split('=')[1];
//   return cookieValue || '';
// }

function ProductRegistration() {
  const [productName, setProductName] = useState('');
  const [productSection, setProductSection] = useState(0);
  const [productPrice, setProductPrice] = useState('');
  const [productMetricUnity, setProductMetricUnity] = useState('');
  const [productAvailibility, setProductAvailibility] = useState('');

  const [suggestions, setSuggestions] = useState([]);

  useEffect(() => {
    if (productName.length > 3) {
      fetch(`http://127.0.0.1:8000/api/products/?search=${productName}`)
        .then((response) => {
          if (!response.ok) {
            throw new Error('Failed to fetch product suggestions');
          }
          return response.json();
        })
        .then((data) => {
          setSuggestions(data);
        })
        .catch((err) => {
          console.error('Error fetching product suggestions: ', err);
          setSuggestions([]);
        });
    } else {
      setSuggestions([]);
    }

  }, [productName]);

  const registerProduct = (event) => {
    event.preventDefault();


    fetch('http://127.0.0.1:8000/cadastrar_produto/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCSRFToken(),
      },
      // credentials: 'include',
      body: JSON.stringify({
        productNameInput: productName.trim(),
        productSectionInput: Number(productSection),
        productPriceInput: productPrice,
        productMetricUnityInput: productMetricUnity.trim(),
        productAvailableInput: productAvailibility,
      }),
    })
    .then((response) => {
      if (!response.ok) {
        throw new Error('Failed to register the product');
      }
      return response.json();
    
    })
    .then((data) => {
      console.log('Product registered succesfully: ', data);
      alert('Product registered succesfully!');
    
    })
    .catch((error) => {
      console.error('Error registering the product: ', error);
    });
  };

  return (
    <div className="flex items-center justify-center min-h-screen p-4">
      <div className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary">
        <div className="flex items-center justify-center mb-5">
          <img className='w-20 mr-4' src={logo} alt="" />
          <h2 className="mb-6 text-2xl font-semibold text-center left-40">Products Registration</h2>
        </div>
        <form action="" className="grid grid-cols-1 gap-6">
          {/* # divide the product name into three parts (name, brand, quantity) */}
          {/* Product Name */}
          <div>
            <label htmlFor="productName" className="block text-sm font-semibold text-white">Product Name</label>
            <input
              type="text"
              id="productName"
              name="productName" required
              className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productName}
              list='productsSuggestions'
              onChange={(e) => setProductName(e.target.value)}
            />
            <datalist id='productsSuggestions'>
              {suggestions.map((product, index) => (
                <option key={index} value={product.nome}>{product.nome}</option>
              ))}
            </datalist>
          </div>

          {/* Product Section */}
          <div>
            <label htmlFor="productSection" className="block text-sm font-semibold text-white">Product Section</label>
            <input
              type="number"
              id="productSection" name="productSection" required className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productSection}
              onChange={(e) => setProductSection(e.target.value)}
            />
          </div>
          {/* Product Price */}
          <div>
            <label htmlFor="productPrice" className="block text-sm font-semibold text-white">Product Price</label>
            <input
              type="text"
              id="productPrice" name="productPrice" required className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productPrice}
              onChange={(e) => setProductPrice(e.target.value)}
            />
          </div>
          {/* Prouct Metric Unity */}
          <div>
            <label htmlFor="productMetricUnity" className="block text-sm font-semibold text-white">Product Metric Unity</label>
            <input type="text" id="productMetricUnity"
              name="ProductRegistration" required className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
              value={productMetricUnity}
              onChange={(e) => setProductMetricUnity(e.target.value)}
            />
          </div>
          {/* Product Availability */}
          <div>
            <label htmlFor="productAvailability" className="block text-sm font-semibold text-white">Product Availability</label>
            <div className="flex">
              <label htmlFor="productAvailability" className="block mr-3 text-sm font-medium text-white">Yes</label>
              <input
                type="radio"
                id="productAvailability"
                name="productAvailability" 
                value="Yes" 
                onChange={(e) => setProductAvailibility(e.target.value)}
                required 
                />
            </div>
            <div className="flex">
              <label htmlFor="productAvailability" className="block mr-3 text-sm font-medium text-white">No&nbsp;&nbsp;</label>
              <input
                type="radio"
                id="productAvailability"
                name="productAvailability"
                value="No" 
                onChange={(e) => setProductAvailibility(e.target.value)}
                required 
                />
            </div>
          </div>

          <button className="px-4 py-2 mt-4 font-bold text-black transition duration-200 bg-yellow-400 rounded-lg hover:bg-yellow-300" type="submit" onClick={registerProduct}>Register</button>
        </form>
      </div>
    </div>
  )
}

export default ProductRegistration