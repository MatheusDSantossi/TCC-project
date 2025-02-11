import { useEffect, useState } from "react";
import logo from "../assets/logo.svg";
import AxiosInstance from "./AxiosInstance";

import ReactiveButton from "reactive-button";

import { Dialog } from "@mui/material";
import { DialogTitle } from "@mui/material";
import { DialogContent } from "@mui/material";

function ProductEdit({ open, onClose, product, onSave }) {
  const [buttonState, setButtonState] = useState("idle");

  const [formData, setFormData] = useState({
    productId: 0,
    productName: "",
    productBrand: "",
    productQuantityFormat: "",
    productSection: 0,
    productPrice: "",
    productMetricUnity: "",
    productAvailability: "",
    productSynonyms: "",
  });

  useEffect(() => {
    console.log("Received product prop: ", product);

    if (product) {
      console.log("Received product prop: ", product.brand);
      setFormData({
        productId: product.product_id || 0,
        productName: product.name || "",
        productBrand: product.brand || "",
        productQuantityFormat: product.quantity || "",
        productSection: product.secao_produto || 0,
        // productPrice: product.price,
        // productMetricUnity: product.metric_unity,
        // productAvailability: product.availability,
        // productSynonyms: product.synonyms,
      });
    }
  }, [product]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      className="flex items-center justify-center min-h-screen p-4 bg-tertiary"
    >
      <div className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary">
        <div className="flex items-center justify-center mb-5">
          <img className="w-20 mr-4" src={logo} alt="" />
          <DialogTitle className="mb-6 text-2xl font-semibold text-center text-white left-40">
            Atualização de Produtos
          </DialogTitle>
        </div>
        <DialogContent>
          {product ? (
            <form onSubmit={handleSubmit}>
              {/* # divide the product name into three parts (name, brand, quantity) */}
              {/* Product Name */}
              <div className="mb-5">
                <label
                  htmlFor="productName"
                  className="block text-sm font-semibold text-white"
                >
                  Nome do Produto
                </label>
                <input
                  type="text"
                  id="productName"
                  name="productName"
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productName}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productName: e.target.value,
                    }))
                  }
                />
              </div>
              <div className="mb-5">
                <label
                  htmlFor="productBrand"
                  className="block text-sm font-semibold text-white"
                >
                  Marca do Produto
                </label>
                <input
                  type="text"
                  id="productBrand"
                  name="productBrand"
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productBrand}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productBrand: e.target.value,
                    }))
                  }
                />
              </div>
              <div className="mb-5">
                <label
                  htmlFor="productBrand"
                  className="block text-sm font-semibold text-white"
                >
                  Formato Quantidade do Produto
                </label>
                <input
                  type="text"
                  id="productQuantityFormat"
                  name="productQuantityFormat"
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productQuantityFormat}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productQuantityFormat: e.target.value,
                    }))
                  }
                />
              </div>
              <div className="mb-5">
                <label
                  htmlFor="productSection"
                  className="block text-sm font-semibold text-white"
                >
                  Seção do Produto
                </label>
                <input
                  type="text"
                  id="productSection"
                  name="productSection"
                  className="w-full p-2 mt-1 text-white border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.productSection}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      productSection: e.target.value,
                    }))
                  }
                />
              </div>

              <ReactiveButton
                className="text-primary hover:bg-yellow-300"
                buttonState={buttonState}
                color="yellow"
                type={"submit"}
                loadingText="Carregando..."
                successText={"Atualizado com sucesso"}
                idleText={"Atualizar produto"}
                animation={true}
                rounded={true}
                width="120"
                height="40"
                size="large"
                // onClick={registerProduct}
              />
            </form>
          ) : (
            <div className="text-center text-gray-400">
              Carregando os produtos...
            </div>
          )}
          {/* )} */}
        </DialogContent>
      </div>
    </Dialog>
  );
}

export default ProductEdit;
