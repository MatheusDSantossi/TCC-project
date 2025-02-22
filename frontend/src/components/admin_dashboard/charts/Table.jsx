import Box from "@mui/material/Box";
import { DataGrid } from "@mui/x-data-grid";
import { IconButton } from "@mui/material";
import { MdEdit } from "react-icons/md";
import { useState } from "react";
import AxiosInstance from "../../AxiosInstance";
import ProductEdit from "../../ProductEdit";

export default function ProductsTable({ products_data, onProductUpdate }) {
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);

  const handleEditClick = (product) => {
    console.log("Product: ", product);
    setSelectedProduct(product);
    setEditModalOpen(true);
  };

  const handleSave = async (updatedProduct) => {
    try {
      const response = await AxiosInstance.put(
        `update-product/${updatedProduct.productId}/`,
        {
          productName: updatedProduct.productName,
          productBrand: updatedProduct.productBrand,
          productQuantityFormat: updatedProduct.productQuantityFormat,
          productSection: updatedProduct.productSection,
          productPrice: updatedProduct.productPrice,
          productMetricUnity: updatedProduct.productMetricUnity,
          productAvailability: updatedProduct.productAvailability,
          productLocalSynonyms: updatedProduct.productLocalSynonyms,
          // productSynonyms: updatedProduct.productSynonyms,
        }
      );
      console.log("Product updated successfully: ", updatedProduct);

      if (response.status === 201 || response.status === 200) {
        onProductUpdate();
      }
    } catch (error) {
      console.error("Error updating product: ", error);
    }
  };

  const handleRefreshTable = () => {
    console.log("Entered to refresh table");
    onProductUpdate();
  };

  const handleClose = () => {
    // setLoading(false);
    setEditModalOpen(false);
  };

  const columns = [
    {
      field: "product_id",
      headerName: <strong>ID</strong>,
      width: 90,
      minWidth: 80,
    },
    {
      field: "name",
      headerName: <strong>Produto</strong>,
      minWidth: 200,
      flex: 2,
    },
    {
      field: "brand",
      headerName: <strong>Marca</strong>,
      minWidth: 150,
      flex: 1,
    },
    {
      field: "quantity",
      headerName: <strong>Formato Qt.</strong>,
      minWidth: 180,
      flex: 1,
    },
    {
      field: "secao_produto",
      headerName: <strong>Seção</strong>,
      minWidth: 100,
      flex: 0.5,
    },
    {
      field: "preco_venda",
      headerName: <strong>Preço</strong>,
      minWidth: 120,
      flex: 1,
    },
    {
      field: "actions",
      headerName: <strong>Ações</strong>,
      minWidth: 100,
      flex: 0.5,
      sortable: false,
      renderCell: (params) => (
        <IconButton
          onClick={() => handleEditClick(params.row)}
          size="small"
          color="primary"
        >
          <MdEdit color="#5D76D3" />
        </IconButton>
      ),
    },
  ];

  // const rows = products_data;

  let rows = [];

  const products_dict = () => {
    for (const product of products_data) {
      rows.push(product);
    }

    return rows;
  };

  console.log("Product dict: ", products_dict()[0]);

  return (
    <div className="flex items-center justify-center w-full overflow-x-auto">
      <Box
        sx={{
          width: "100%",
          height: "calc(100 - 100px)",
          padding: 2,
          "& .MuiDataGrid-root": {
            minWidth: 800, // Minimum width before scrolling
          },
        }}
      >
        <DataGrid
          rows={products_data}
          columns={columns}
          getRowId={(row) => row.product_id}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 10,
              },
            },
          }}
          pageSizeOptions={[10]}
          disableRowSelectionOnClick
          sx={{
            boxShadow: 2,
            border: 2,
            borderColor: '#5D76D3',
            "& .MuiDataGrid-cell": {
              backgroundColor: "#F4F4F4FF",
              whiteSpace: "normal",
              lineHeight: "1.2",
              padding: "8px",
            },
            "& .MuiDataGrid-columnHeaderTitle": {
              fontWeight: "bold",
            },
            '& .MuiDataGrid-cell:hover': {
              color: '#5D76D3',
            },
          }}
        />
        <ProductEdit
          open={editModalOpen}
          onClose={handleClose}
          product={selectedProduct}
          onSave={handleSave}
          refreshTable={handleRefreshTable}
        />
      </Box>
    </div>
  );
}
