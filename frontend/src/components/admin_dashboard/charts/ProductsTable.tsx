import Box from "@mui/material/Box";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import { IconButton } from "@mui/material";
import { MdEdit } from "react-icons/md";
import { useMemo, useState } from "react";
import AxiosInstance from "../../AxiosInstance";
import ProductEdit from "../../productsRegistration/ProductEdit";
import CustomLoadingOverlay, { CustomNoRowsOverlay } from "./components/CustomLoadingOverlay";

export default function ProductsTable({
  products_data,
  onProductUpdate,
  isLoading,
}) {
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState(null);

  const handleEditClick = (product) => {
    // console.log("Product: ", product);
    setSelectedProduct(product);
    setEditModalOpen(true);
  };

  const handleSave = async (updatedProduct) => {
    try {
      const response = await AxiosInstance.put(
        `api/products/${updatedProduct.productId}/`,
        {
          name: updatedProduct.productName,
          brand: updatedProduct.productBrand,
          quantity: updatedProduct.productQuantityFormat,
          section: updatedProduct.productSection,
          price: updatedProduct.productPrice,
          unit: updatedProduct.productMetricUnity,
          stock: updatedProduct.productAvailability,
          synonymous: updatedProduct.productLocalSynonyms,
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

  const columns = useMemo(
    () => [
      {
        field: "product_id",
        headerName: <strong className="text-base">ID</strong>,
        width: 90,
        minWidth: 80,
      },
      {
        field: "name",
        headerName: <strong className="text-base">Produto</strong>,
        minWidth: 200,
        flex: 2,
      },
      {
        field: "brand",
        headerName: <strong className="text-base">Marca</strong>,
        minWidth: 150,
        flex: 1,
      },
      {
        field: "quantity",
        headerName: <strong className="text-base">Formato Qt.</strong>,
        minWidth: 180,
        flex: 1,
      },
      {
        field: "section",
        headerName: <strong className="text-base">Seção</strong>,
        minWidth: 100,
        flex: 0.5,
      },
      {
        field: "price",
        headerName: <strong className="text-base">Preço</strong>,
        minWidth: 120,
        flex: 1,
      },
      {
        field: "actions",
        headerName: <strong className="text-base">Ações</strong>,
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
    ],
    []
  );

  // const rows = products_data;

  let rows = [];

  const products_dict = () => {
    for (const product of products_data) {
      rows.push(product);
    }

    return rows;
  };

  // console.log("Product dict: ", products_dict()[0]);

  return (
    <div className="flex w-full h-[730px]">
      <Box
        sx={{
          position: "sticky",
          zIndex: 1,
          // height: "50%",
          // minHeight: 340,
          display: "flex",
          flexDirection: "column",
          width: "100%",
          pl: 1.5,
          pr: 1,
          py: 1,
          bottom: 0,
          border: "1px solid",
          borderColor: "divider",
          // boxShadow: "none",
          borderRadius: 1,
          // backgroundColor: "background.paper",
          "& .MuiDataGrid-root": {
            minWidth: 100, // Minimum width before scrolling
          },
        }}
      >
        <DataGrid
          rows={products_data ?? []}
          disableVirtualization
          columns={columns}
          getRowId={(row) => row.product_id}
          loading={!!isLoading}
          components={{
            LoadingOverlay: CustomLoadingOverlay,
            NoRowsOverlay: CustomNoRowsOverlay,
          }}
          componentsProps={{
            noRowsOverlay: {
              // pass a dynamic message based on your app state
              message: isLoading ? "Carregando..." : "Nenhum produto cadastrado.",
            },
          }}
          initialState={{
            pagination: {
              paginationModel: {
                pageSize: 15, // HERE I LIMIT THE QUANTITY OF ITEMS
              },
            },
          }}
          pageSizeOptions={[10]}
          disableRowSelectionOnClick
          density="comfortable"
          sx={{
            fontSize: "14px",
            boxShadow: 2,
            border: 2,
            borderColor: "#5D76D3",
            backgroundColor: "#FFFFFFFF",
            "& .MuiDataGrid-cell": {
              backgroundColor: "#F4F4F4FF",
              whiteSpace: "normal",
              lineHeight: "1.2",
              padding: "8px",
            },
            "& .MuiDataGrid-columnHeaderTitle": {
              fontWeight: "bold",
            },
            "& .MuiDataGrid-cell:hover": {
              color: "#5D76D3",
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
