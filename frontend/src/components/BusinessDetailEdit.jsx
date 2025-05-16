import { useEffect, useState } from "react";
import logo from "../assets/logo.svg";

import { Dialog } from "@mui/material";
import { DialogTitle } from "@mui/material";
import { DialogContent } from "@mui/material";

import ReactiveButton from "reactive-button";
import SynonymDetails from "./SynonymsDetails";
import AxiosInstance from "./AxiosInstance";

import { MdOutlineClose } from "react-icons/md";

function BusinessDetailEdit({ open, onClose, businessDetail, onSave, refreshTable }) {
  const [updateButtonState, setUpdateButtonState] = useState("idle");
  const [deleteButtonState, setDeleteButtonState] = useState("idle");
  const [resetKey, setResetKey] = useState(0);

  const [formData, setFormData] = useState({
    key: "",
    value: "",
  });

  useEffect(() => {
    console.log("Received business detail prop: ", businessDetail);
    // setButtonState("Carregando...");
    if (businessDetail) {
      console.log("Received business detail prop: ", businessDetail.key);
      setFormData({
        id: businessDetail.id || 0,
        key: businessDetail.key || "",
        value: businessDetail.value || "",
      });
      setResetKey((prev) => prev + 1);
      // setButtonState("Sucesso");
    }
  }, [businessDetail]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onSave(formData);
    onClose();
  };

  const handleDeleteBusinessDetail = async () => {
    onClose();
    try {
      const response = await AxiosInstance.delete(`delete-business-detail/${formData.id}/`);

      refreshTable();
      
      console.log(`Business detail ${formData.key} deleted succesfuly`, response.data);
    } catch (error) {
      console.error("Error deleting product: ", error);
    }
    
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      className="flex items-center justify-center min-h-screen p-4 bg-tertiary"
    >
        <div className="w-full max-w-2xl p-8 rounded-lg shadow-xl bg-primary">
          <MdOutlineClose className="text-3xl text-white cursor-pointer hover:text-secondary" onClick={onClose} />
          <div className="flex items-center justify-center">
            <img className="w-20 mr-4" src={logo} alt="" />
            <DialogTitle className="text-2xl font-semibold text-center text-white ">
              Atualização de Informações do Business
            </DialogTitle>
          </div>
          <DialogContent>
            <form onSubmit={handleSubmit}>
              {/* # divide the product name into three parts (name, brand, quantity) */}
              {/* Product Name */}
              <div className="mb-6">
                <label
                  htmlFor="businessDetailTitle"
                  className="block mb-2 text-sm font-semibold text-white"
                >
                  Título da Informação
                </label>
                <input
                  type="text"
                  id="businessDetailTitle"
                  name="businessDetailTitle"
                  className="w-full p-3 text-white border-gray-300 rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
                  value={formData?.key}
                  onChange={(e) =>
                    setFormData((prev) => ({
                      ...prev,
                      key: e.target.value,
                    }))
                  }
                />
              </div>
              {/* <div className="grid grid-cols-2 gap-4 mb-6"> */}
                <div>
                  <label
                    htmlFor="businessDetailDescription"
                    className="block mb-2 text-sm font-semibold text-white"
                  >
                    Descrição da Informação
                  </label>
                  <textarea
                    id="businessDetailDescription"
                    name="businessDetailDescription"
                    required
                    className="w-full p-4 mt-1 text-white rounded-lg bg-input focus:outline-none focus:ring-2 focus:ring-yellow-400"
                    value={formData?.value}
                    onChange={(e) =>
                      setFormData((prev) => ({
                        ...prev,
                        value: e.target.value,
                      }))
                    }
                    placeholder="Esse serviço é..."
                    rows={4}
                    cols={50}
                  />
                </div>

              {/* Buttons */}
              <div className="flex gap-4 mt-8 ml-8">
                <ReactiveButton
                  className="text-primary hover:bg-yellow-300"
                  buttonState={updateButtonState}
                  color="yellow"
                  type={"submit"}
                  loadingText="Carregando..."
                  successText={"Atualizado com sucesso"}
                  idleText={"Atualizar"}
                  animation={true}
                  rounded={true}
                  width="120"
                  height="40"
                  size="large"
                />
                <ReactiveButton
                  className="ml-20 text-primary hover:bg-yellow-300"
                  buttonState={deleteButtonState}
                  color="yellow"
                  onClick={handleDeleteBusinessDetail}
                  loadingText="Carregando..."
                  successText={"Excluído com sucesso"}
                  idleText={"Excluir"}
                  animation={true}
                  rounded={true}
                  width="120"
                  height="40"
                  size="large"
                />
              </div>
            </form>
          </DialogContent>
        </div>
    </Dialog>
  );
}

export default BusinessDetailEdit;
