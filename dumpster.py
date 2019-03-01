
#    # --------------------------------------------------------------------------------
#     # -----Loss function--------------------------------------------------------------
#     # --------------------------------------------------------------------------------
#     def loss_function_all(self, X):
#         loss = 0
#         n_inv = len(X) ** -1
#         for point in X:
#             loss += n_inv * self.loss_function_single_point(point)
#         return loss

#     # --------------------------------------------------------------------------------
#     # -----Update rules for the weights-----------------------------------------------
#     # --------------------------------------------------------------------------------

#     # --------------------------------------------------------------------------------
#     def hidden_weights_change_point(self, objects, hidden_derivs, value_E):
#         point, u, sigma_derivs, N, DN, D2N = objects
#         dH_N, dH_DN, dH_D2N = hidden_derivs
#         H = np.zeros((self.hidden_dim, self.input_dim)).astype(dtype="float64")
#         for m in range(self.hidden_dim):
#             for p in range(self.input_dim):
#                 for j in range(self.visible_dim):
#                     for i in range(self.input_dim):
#                         H[m, p] += dH_D2N[j, i, m, p] * np.flip(N, axis=0)[j]
#                         H[m, p] += D2N[j, i] * \
#                             np.flip(dH_N, axis=0)[j, 0, m, p]
#                         H[m, p] += 2 * dH_DN[j, i, m, p] * \
#                             np.flip(DN, axis=0)[j, i]
#                     H[m, p] += value_E * dH_N[j, 0, m, p] * \
#                         np.flip(N, axis=0)[j]
#         return H

#     # --------------------------------------------------------------------------------
#     # --------------------------------------------------------------------------------

#     # --------------------------------------------------------------------------------
#     def visible_weights_change_point(self, objects, visible_derivs, value_E):
#         point, u, sigma_derivs, N, DN, D2N = objects
#         dV_N, dV_DN, dV_D2N = visible_derivs
#         V = np.zeros((self.visible_dim, self.hidden_dim)
#                      ).astype(dtype="float64")
#         for m in range(self.visible_dim):
#             for p in range(self.hidden_dim):
#                 for j in range(self.visible_dim):
#                     for i in range(self.input_dim):
#                         V[m, p] += dV_D2N[j, i, m, p] * np.flip(N, axis=0)[j]
#                         V[m, p] += D2N[j, i] * \
#                             np.flip(dV_N, axis=0)[j, 0, m, p]
#                         V[m, p] += 2 * dV_DN[j, i, m, p] * \
#                             np.flip(DN, axis=0)[j, i]
#                     V[m, p] += value_E * dV_N[j, 0, m, p] * \
#                         np.flip(N, axis=0)[j]
#         return V

#     # --------------------------------------------------------------------------------
#     def bias_change_point(self, objects, bias_derivs, value_E):
#         point, u, sigma_derivs, N, DN, D2N = objects
#         db_N, db_DN, db_D2N = bias_derivs
#         b = np.zeros(shape=(self.hidden_dim, 1))
#         for m in range(self.hidden_dim):
#             for j in range(self.visible_dim):
#                 for i in range(self.input_dim):
#                     b[m] += db_D2N[j, i, m] * np.flip(N, axis=0)[j]
#                     b[m] += D2N[j, i] * np.flip(db_N, axis=0)[j, 0, m]
#                     b[m] += 2 * db_DN[j, i, m] * np.flip(DN, axis=0)[j, i]
#                 b[m, 0] += value_E * db_N[j, 0, m] * np.flip(N, axis=0)[j]
#         return b
